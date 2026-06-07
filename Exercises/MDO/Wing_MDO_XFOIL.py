import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from warnings import warn

from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import subprocess
import os
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator # Keep Pchip for baseline loading if preferred
import time

from OMADS import mads

# Import NURBS library
from geomdl import NURBS
from geomdl import utilities
# from geomdl.helpers import get_curve_points # Helper to evaluate curve

# --- 1. Configuration and Global Parameters ---
WING_SPAN = 10.0  # meters
CHORD_ROOT = 2.0  # meters
CHORD_TIP = 1.0   # meters
MAX_STRESS_ALLOWED = 30e6  # Pa (e.g., 30 MPa for aluminum)
MAX_DEFLECTION_ALLOWED = 0.5  # meters (e.g., 5% of span)
AIR_DENSITY = 1.225  # kg/m^3
FLIGHT_SPEED = 100.0  # m/s
ANGLE_OF_ATTACK = 0.0 # degrees, simplified for initial example
NUM_SPAN_SECTIONS = 3 # Reduced for faster initial testing with more DVs per section

# Material properties for beam analysis (e.g., Aluminum)
YOUNG_MODULUS = 70e9 # Pa
POISSON_RATIO = 0.33

# --- NURBS Parameters (Revised for 3 Curves) ---
NURBS_DEGREE = 3 # Cubic NURBS

# Number of INTERNAL control points for each of the three segments
# (excluding fixed LE/TE points and junction points)
N_LE_INTERNAL_CPS = 0 # No internal CPs for LE curve, as they are now fixed
N_UPPER_INTERNAL_CPS = 3 # Internal CPs for the upper curve
N_LOWER_INTERNAL_CPS = 3 # Internal CPs for the lower curve

# Total design variables per section:
# N_LE_INTERNAL_CPS (y-coords) + N_UPPER_INTERNAL_CPS (y-coords) + N_LOWER_INTERNAL_CPS (y-coords)
# The LE junction point is now fixed, so no DV for it.
NUM_DVS_PER_SECTION = N_UPPER_INTERNAL_CPS + N_LOWER_INTERNAL_CPS

# --- Define fixed x-coordinates for NURBS internal control points ---
# These are the x/c locations for the *internal* control points for each segment.
# We'll define the junction point at a specific x-coordinate.
X_LE_JUNCTION = 0.05 # x-coordinate where LE curve transitions to upper/lower

ENABLE_LIVE_PLOT = False # Set to True to enable live plotting during optimization

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "post", "Wing-design_run", "wing_eval_log.csv")


@dataclass
class PredictionSummaryWing:
    objective: float
    g1: float
    g2: float
    g3: float
    prob_infeasible: float
    prob_no_improve: float


def _poly_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    cross = [x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x))]
    return np.concatenate(([1.0], x, x**2, np.asarray(cross, dtype=np.float64)))


def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -40.0, 40.0))
    return float(1.0 / (1.0 + np.exp(-z)))


class WingBayesianEnsembleSurrogate:
    def __init__(
        self,
        csv_path: str,
        initial_designs: int = 20,
        retrain_every_simulated: int = 5,
        ensemble_size: int = 5,
        hidden_size: int = 64,
        epochs: int = 250,
        lr: float = 1e-3,
        max_predicted_streak: int = 4,
        improve_margin: float = 1e-4,
        prediction_acceptance_threshold: float = 0.65,
        seed: int = 0,
    ):
        self.csv_path = csv_path
        self.initial_designs = int(initial_designs)
        self.retrain_every_simulated = int(retrain_every_simulated)
        self.ensemble_size = int(ensemble_size)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.max_predicted_streak = int(max_predicted_streak)
        self.improve_margin = float(improve_margin)
        self.prediction_acceptance_threshold = float(prediction_acceptance_threshold)
        self.rng = np.random.default_rng(seed)

        self.records: List[Dict] = []
        self.df = pd.DataFrame()
        self.trained = False
        self.numpy_models: List[Dict[str, np.ndarray]] = []
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.total_simulated = 0
        self.simulated_since_retrain = 0
        self.predicted_streak = 0
        self.best_feasible_objective = np.inf
        self.y_std_floor: Optional[np.ndarray] = None

    def _should_predict(self) -> bool:
        return self.trained and self.predicted_streak < self.max_predicted_streak

    def _prediction_is_acceptable(self, pred: PredictionSummaryWing) -> bool:
        threshold = self.prediction_acceptance_threshold
        if not np.isfinite(pred.prob_infeasible) and not np.isfinite(pred.prob_no_improve):
            return False
        return (np.isfinite(pred.prob_infeasible) and pred.prob_infeasible >= threshold) or (
            np.isfinite(pred.prob_no_improve) and pred.prob_no_improve >= threshold
        )

    def evaluate(self, x: List[float], simulator_fn: Callable[[List[float]], List]):
        x_np = np.asarray(x, dtype=float)

        if False:#self._should_predict():
            pred = self.predict_with_probabilities(x_np)
            if (
                not np.isfinite(pred.objective)
                or not np.isfinite(pred.g1)
                or not np.isfinite(pred.g2)
                or not np.isfinite(pred.g3)
                or not self._prediction_is_acceptable(pred)
            ):
                return self._simulate_and_record(x_np, simulator_fn, pred)

            c1, c2, c3 = pred.g1, pred.g2, pred.g3
            feasible = c1 <= 0.0 and c2 <= 0.0 and c3 <= 0.0
            self.predicted_streak += 1
            self._append_record(
                x_np,
                pred.objective,
                pred.g1,
                pred.g2,
                pred.g3,
                feasible,
                "predicted",
                pred.prob_infeasible,
                pred.prob_no_improve,
            )
            return [float(pred.objective), [float(c1), float(c2), float(c3)]]

        return self._simulate_and_record(x_np, simulator_fn)

    def _simulate_and_record(
        self,
        x_np: np.ndarray,
        simulator_fn: Callable[[List[float]], List],
        pred: Optional[PredictionSummaryWing] = None,
    ):
        y = simulator_fn(x_np.tolist())
        objective = float(y[0])
        g1, g2, g3 = [float(v) for v in y[1]]
        feasible = g1 <= 0.0 and g2 <= 0.0 and g3 <= 0.0 and np.isfinite(objective)

        self.predicted_streak = 0
        self.total_simulated += 1
        self.simulated_since_retrain += 1
        if feasible and objective < self.best_feasible_objective:
            self.best_feasible_objective = float(objective)

        p_infeasible = np.nan
        p_no_improve = np.nan
        if pred is not None:
            p_infeasible = pred.prob_infeasible
            p_no_improve = pred.prob_no_improve
        elif self.trained:
            pred_after = self.predict_with_probabilities(x_np)
            p_infeasible = pred_after.prob_infeasible
            p_no_improve = pred_after.prob_no_improve

        self._append_record(
            x_np,
            objective,
            g1,
            g2,
            g3,
            feasible,
            "simulated",
            p_infeasible,
            p_no_improve,
        )
        self._maybe_retrain()
        return [objective, [g1, g2, g3]]

    def _append_record(
        self,
        x: np.ndarray,
        objective: float,
        g1: float,
        g2: float,
        g3: float,
        feasible: bool,
        source: str,
        prob_infeasible: float,
        prob_no_improve: float,
    ):
        row = {
            "iter": len(self.records) + 1,
            "objective_min": float(objective),
            "g1": float(g1),
            "g2": float(g2),
            "g3": float(g3),
            "feasible": bool(feasible),
            "source": source,
            "prob_infeasible": float(prob_infeasible) if np.isfinite(prob_infeasible) else np.nan,
            "prob_no_improve": float(prob_no_improve) if np.isfinite(prob_no_improve) else np.nan,
            "best_feasible_objective": float(self.best_feasible_objective)
            if np.isfinite(self.best_feasible_objective)
            else np.nan,
        }
        for i, xi in enumerate(x):
            row[f"x{i}"] = float(xi)

        self.records.append(row)
        self.df = pd.DataFrame(self.records)
        out_dir = os.path.dirname(self.csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.df.to_csv(self.csv_path, index=False)

    def _maybe_retrain(self):
        if self.total_simulated < self.initial_designs:
            return
        if not self.trained:
            self._fit_from_simulated()
            self.trained = bool(self.numpy_models)
            self.simulated_since_retrain = 0
            return
        if self.simulated_since_retrain >= self.retrain_every_simulated:
            self._fit_from_simulated()
            self.trained = bool(self.numpy_models)
            self.simulated_since_retrain = 0

    def _fit_from_simulated(self):
        sim_df = self.df[self.df["source"] == "simulated"].copy()
        sim_df = sim_df.replace([np.inf, -np.inf], np.nan)
        sim_df = sim_df.dropna(subset=["objective_min", "g1", "g2", "g3"]).copy()
        if len(sim_df) < max(self.initial_designs, 4):
            return

        x_cols = sorted([c for c in sim_df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
        y_cols = ["objective_min", "g1", "g2", "g3"]

        x_data = sim_df[x_cols].to_numpy(dtype=np.float32)
        y_data = sim_df[y_cols].to_numpy(dtype=np.float32)

        finite_mask = np.isfinite(x_data).all(axis=1) & np.isfinite(y_data).all(axis=1)
        x_data = x_data[finite_mask]
        y_data = y_data[finite_mask]
        if len(x_data) < max(self.initial_designs, 4):
            return

        # Keep uncertainty from collapsing to near-zero so probabilities stay informative.
        self.y_std_floor = np.maximum(np.nanstd(y_data, axis=0) * 0.05, 1e-3)

        self.x_mean = x_data.mean(axis=0)
        self.x_std = x_data.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x_norm = (x_data - self.x_mean) / self.x_std

        features = np.vstack([_poly_features(row) for row in x_norm])
        ridge = 1e-8
        self.numpy_models = []
        for _ in range(self.ensemble_size):
            sample_idx = self.rng.integers(0, len(features), size=len(features))
            x_boot = features[sample_idx]
            y_boot = y_data[sample_idx]
            xtx = x_boot.T @ x_boot
            coef = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), x_boot.T @ y_boot)
            residuals = y_boot - x_boot @ coef
            variance = residuals.var(axis=0, ddof=1) if len(residuals) > 1 else np.full(y_boot.shape[1], 1e-6)
            variance = np.where(np.isfinite(variance), variance, 1e-6)
            self.numpy_models.append({"coef": coef, "variance": np.maximum(variance, 1e-6)})

    def predict_with_probabilities(self, x: np.ndarray) -> PredictionSummaryWing:
        if not self.trained or not self.numpy_models or self.x_mean is None or self.x_std is None:
            raise RuntimeError("Surrogate is not trained yet.")

        x_in = ((x.astype(np.float32) - self.x_mean) / self.x_std).reshape(1, -1)

        means = []
        vars_ = []
        features = _poly_features(x_in.reshape(-1))
        for model in self.numpy_models:
            means.append(features @ model["coef"])
            vars_.append(model["variance"])

        means = np.asarray(means)
        vars_ = np.asarray(vars_)
        means = np.where(np.isfinite(means), means, np.nan)
        vars_ = np.where(np.isfinite(vars_), vars_, np.nan)

        if np.isnan(means).all() or np.isnan(vars_).all():
            return PredictionSummaryWing(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

        agg_mean = means.mean(axis=0)
        agg_var = (vars_ + means**2).mean(axis=0) - agg_mean**2
        agg_var = np.where(np.isfinite(agg_var), agg_var, 1e-8)
        agg_var = np.maximum(agg_var, 1e-8)
        agg_std = np.sqrt(agg_var)
        if self.y_std_floor is not None:
            agg_std = np.maximum(agg_std, self.y_std_floor)

        obj_mean, g1_mean, g2_mean, g3_mean = [float(v) for v in agg_mean]
        obj_std, g1_std, g2_std, g3_std = [float(v) for v in agg_std]

        n_mc = 2000
        obj_samples = self.rng.normal(obj_mean, obj_std, size=n_mc)
        g1_samples = self.rng.normal(g1_mean, g1_std, size=n_mc)
        g2_samples = self.rng.normal(g2_mean, g2_std, size=n_mc)
        g3_samples = self.rng.normal(g3_mean, g3_std, size=n_mc)
        prob_infeasible = float(np.mean((g1_samples > 0.0) | (g2_samples > 0.0) | (g3_samples > 0.0)))

        if np.isfinite(self.best_feasible_objective):
            prob_no_improve = float(np.mean(obj_samples >= (self.best_feasible_objective - self.improve_margin)))
        else:
            prob_no_improve = np.nan

        eps = 1.0 / (n_mc + 2.0)
        prob_infeasible = float(np.clip(prob_infeasible, eps, 1.0 - eps))
        if np.isfinite(prob_no_improve):
            prob_no_improve = float(np.clip(prob_no_improve, eps, 1.0 - eps))

        return PredictionSummaryWing(obj_mean, g1_mean, g2_mean, g3_mean, prob_infeasible, prob_no_improve)


wing_surrogate: Optional[WingBayesianEnsembleSurrogate] = None


def initialize_wing_surrogate(
    csv_path: str = DEFAULT_CSV_PATH,
    initial_designs: int = 100,
    retrain_every_simulated: int = 50,
    ensemble_size: int = 5,
    hidden_size: int = 64,
    epochs: int = 250,
    lr: float = 1e-3,
    max_predicted_streak: int = 4,
    improve_margin: float = 1e-4,
    prediction_acceptance_threshold: float = 0.65,
    seed: int = 0,
):
    global wing_surrogate
    wing_surrogate = WingBayesianEnsembleSurrogate(
        csv_path=csv_path,
        initial_designs=initial_designs,
        retrain_every_simulated=retrain_every_simulated,
        ensemble_size=ensemble_size,
        hidden_size=hidden_size,
        epochs=epochs,
        lr=lr,
        max_predicted_streak=max_predicted_streak,
        improve_margin=improve_margin,
        prediction_acceptance_threshold=prediction_acceptance_threshold,
        seed=seed,
    )
    _bootstrap_wing_surrogate_from_csv(wing_surrogate)


def _bootstrap_wing_surrogate_from_csv(surrogate: WingBayesianEnsembleSurrogate):
    if not os.path.exists(surrogate.csv_path):
        return
    try:
        hist = pd.read_csv(surrogate.csv_path)
    except Exception as exc:
        warn(f"Unable to read wing surrogate history CSV at {surrogate.csv_path}: {exc}")
        return

    if hist.empty:
        return

    required = {"source", "objective_min", "feasible", "g1", "g2", "g3"}
    if not required.issubset(set(hist.columns)):
        warn("Existing wing surrogate history CSV has incompatible schema; starting fresh in-memory history.")
        return

    surrogate.df = hist.copy()
    surrogate.records = hist.to_dict(orient="records")
    sim_df = surrogate.df[surrogate.df["source"] == "simulated"].copy()
    surrogate.total_simulated = int(len(sim_df))
    surrogate.simulated_since_retrain = int(len(sim_df) % max(1, surrogate.retrain_every_simulated))
    feasible_sim = sim_df[sim_df["feasible"] == True]
    surrogate.best_feasible_objective = float(feasible_sim["objective_min"].min()) if not feasible_sim.empty else np.inf

    if surrogate.total_simulated >= surrogate.initial_designs:
        try:
            surrogate._fit_from_simulated()
            surrogate.trained = bool(surrogate.numpy_models)
        except Exception as exc:
            warn(f"Wing surrogate warm-start training from CSV failed: {exc}")
            surrogate.trained = False


def evaluate_wing(design_variables):
    global wing_surrogate
    if wing_surrogate is None:
        initialize_wing_surrogate()

    y = wing_surrogate.evaluate(design_variables, eval_opt)
    row_idx = wing_surrogate.df.index[-1]
    last_row = wing_surrogate.df.iloc[-1]
    pinf = last_row["prob_infeasible"]
    pni = last_row["prob_no_improve"]

    if pd.isna(pinf):
        sim_df = wing_surrogate.df[wing_surrogate.df["source"] == "simulated"]
        if not sim_df.empty:
            c_scale = float(
                np.nanmean(
                    [
                        max(np.nanstd(sim_df["g1"]), 1e-3),
                        max(np.nanstd(sim_df["g2"]), 1e-3),
                        max(np.nanstd(sim_df["g3"]), 1e-3),
                    ]
                )
            )
        else:
            c_scale = 1e-2

        c_probs = [_sigmoid(float(ci) / c_scale) for ci in y[1]]
        pinf = float(1.0 - np.prod([1.0 - p for p in c_probs]))

    if pd.isna(pni):
        sim_df = wing_surrogate.df[wing_surrogate.df["source"] == "simulated"]
        if not sim_df.empty:
            feasible_sim = sim_df[sim_df["feasible"] == True]
            ref_obj = float(feasible_sim["objective_min"].min()) if not feasible_sim.empty else float(sim_df["objective_min"].min())
            obj_scale = float(max(np.nanstd(sim_df["objective_min"]), abs(ref_obj) * 0.02, 1e-3))
            obj_margin = float(y[0]) - (ref_obj - wing_surrogate.improve_margin)
            pni = _sigmoid(obj_margin / obj_scale)
        else:
            pni = np.nan

    wing_surrogate.df.loc[row_idx, "prob_infeasible"] = pinf
    wing_surrogate.df.loc[row_idx, "prob_no_improve"] = pni
    wing_surrogate.records[-1]["prob_infeasible"] = float(pinf) if pd.notna(pinf) else np.nan
    wing_surrogate.records[-1]["prob_no_improve"] = float(pni) if pd.notna(pni) else np.nan
    wing_surrogate.df.to_csv(wing_surrogate.csv_path, index=False)

    print(
        f"Wing eval: source={last_row['source']}, obj={y[0]:.6f}, g1={y[1][0]:.6f}, g2={y[1][1]:.6f}, g3={y[1][2]:.6f}, "
        f"P(infeasible)={pinf if pd.notna(pinf) else float('nan'):.4f}, P(no-improve)={pni if pd.notna(pni) else float('nan'):.4f}"
    )
    return y

def get_nurbs_internal_x_coords_for_segment(num_internal_cps, x_start, x_end):
    if num_internal_cps <= 0:
        return np.array([])
    # Generate points using a cosine distribution between x_start and x_end
    x_points_raw = (1 - np.cos(np.linspace(0, np.pi, num_internal_cps + 2))) / 2
    x_points_raw = x_points_raw[1:-1] # Exclude 0 and 1
    x_points = x_start + (x_end - x_start) * x_points_raw
    return np.unique(np.sort(x_points))

# X-coordinates for internal CPs for each segment
# LE internal CPs are no longer design variables, but we still define the fixed CPs for the LE curve
X_INTERNAL_UPPER_CPS = get_nurbs_internal_x_coords_for_segment(N_UPPER_INTERNAL_CPS, X_LE_JUNCTION, 1.0)
X_INTERNAL_LOWER_CPS = get_nurbs_internal_x_coords_for_segment(N_LOWER_INTERNAL_CPS, X_LE_JUNCTION, 1.0)

# --- Global definition for NUM_DVS_TOTAL ---
NUM_DVS_TOTAL = NUM_SPAN_SECTIONS * NUM_DVS_PER_SECTION

# --- Helper Function to Load Airfoil Data ---
# This function now returns the processed upper and lower baseline data separately
def load_airfoil_data(filepath, num_points=200):
    """
    Loads airfoil coordinates from a .dat file and interpolates to a fixed number of points.
    Returns the raw, sorted, and unique x,y for the full airfoil, and for upper/lower surfaces.
    """
    try:
        data = np.genfromtxt(filepath, skip_header=1)
        x_raw_full, y_raw_full = data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error loading airfoil data from {filepath}: {e}")
        print("Using embedded RAE2822 data for demonstration.")
        # Embedded RAE2822 data (simplified, for demonstration if file not found)
        x_raw_full = np.array([
            1.000000, 0.998900, 0.995600, 0.990100, 0.982400, 0.972500, 0.960500, 0.946300,
            0.930000, 0.911600, 0.891100, 0.868500, 0.843900, 0.817300, 0.788800, 0.758400,
            0.726200, 0.692300, 0.656700, 0.619500, 0.580800, 0.540700, 0.499300, 0.456700,
            0.413100, 0.368600, 0.323300, 0.277400, 0.231000, 0.184300, 0.137400, 0.090500,
            0.043800, 0.000000, # Leading edge
            0.043800, 0.090500, 0.137400, 0.184300, 0.231000, 0.277400, 0.323300, 0.368600,
            0.413100, 0.456700, 0.499300, 0.540700, 0.580800, 0.619500, 0.656700, 0.692300,
            0.726200, 0.758400, 0.788800, 0.817300, 0.843900, 0.868500, 0.891100, 0.911600,
            0.930000, 0.946300, 0.960500, 0.972500, 0.990100, 0.995600, 0.998900, 1.000000
        ])
        y_raw_full = np.array([
            0.000000, 0.000600, 0.001700, 0.003300, 0.005300, 0.007700, 0.010500, 0.013700,
            0.017300, 0.021200, 0.025500, 0.030000, 0.034800, 0.039800, 0.044900, 0.050000,
            0.055000, 0.059800, 0.064200, 0.068200, 0.071600, 0.074400, 0.076500, 0.077800,
            0.078300, 0.077800, 0.076300, 0.073700, 0.069900, 0.064700, 0.058100, 0.049900,
            0.039900, 0.000000, # Leading edge
            -0.039900, -0.049900, -0.058100, -0.064700, -0.069900, -0.073700, -0.076300, -0.077800,
            -0.078300, -0.077800, -0.076500, -0.074400, -0.071600, -0.068200, -0.064200, -0.059800,
            -0.055000, -0.050000, -0.044900, -0.039800, -0.034800, -0.030000, -0.025500, -0.021200,
            -0.017300, -0.013700, -0.184300, -0.231000, -0.277400, -0.323300, -0.368600,
            0.000000
        ])

    # Process full raw data to ensure it's sorted by x and unique
    sort_indices_full = np.argsort(x_raw_full)
    x_sorted_full = x_raw_full[sort_indices_full]
    y_sorted_full = y_raw_full[sort_indices_full]
    _, unique_indices_full = np.unique(x_sorted_full, return_index=True)
    x_unique_full = x_sorted_full[unique_indices_full]
    y_unique_full = y_sorted_full[unique_indices_full]

    # Split into upper and lower surfaces based on y-values at common x-points
    # A simple heuristic for splitting: positive y for upper, negative for lower.
    # This assumes a standard airfoil shape and (0,0) at LE.
    
    # Upper surface points (from LE to TE)
    x_upper_orig = x_unique_full[y_unique_full >= 0]
    y_upper_orig = y_unique_full[y_unique_full >= 0]
    
    # Lower surface points (from LE to TE)
    x_lower_orig = x_unique_full[y_unique_full <= 0]
    y_lower_orig = y_unique_full[y_unique_full <= 0]

    # Ensure x values are strictly increasing for PchipInterpolator
    # (already handled by np.unique and argsort, but good to double check)
    sort_idx_upper = np.argsort(x_upper_orig)
    x_upper_orig = x_upper_orig[sort_idx_upper]
    y_upper_orig = y_upper_orig[sort_idx_upper]

    sort_idx_lower = np.argsort(x_lower_orig)
    x_lower_orig = x_lower_orig[sort_idx_lower]
    y_lower_orig = y_lower_orig[sort_idx_lower]

    # --- Crucial check: Ensure at least 2 elements for interpolator ---
    # If not enough points, provide a minimal valid set to prevent PchipInterpolator errors
    if len(x_upper_orig) < 2:
        print("Warning: Not enough unique points for baseline upper surface. Using simplified data.")
        x_upper_orig = np.array([0.0, 1.0])
        y_upper_orig = np.array([0.0, 0.0]) # Flat line to avoid errors
    if len(x_lower_orig) < 2:
        print("Warning: Not enough unique points for baseline lower surface. Using simplified data.")
        x_lower_orig = np.array([0.0, 1.0])
        y_lower_orig = np.array([0.0, 0.0]) # Flat line to avoid errors

    # Return all necessary baseline data
    return x_unique_full, y_unique_full, x_upper_orig, y_upper_orig, x_lower_orig, y_lower_orig

# Load the baseline RAE2822 airfoil coordinates once
# IMPORTANT: Update this path to where your rae2822.dat file is located!
RAE2822_X_BASELINE_FULL, RAE2822_Y_BASELINE_FULL, \
RAE2822_X_BASELINE_UPPER, RAE2822_Y_BASELINE_UPPER, \
RAE2822_X_BASELINE_LOWER, RAE2822_Y_BASELINE_LOWER = load_airfoil_data(r"C:\apps\code\Py_Dev\WS-AIMDO\simple_airfoil\CFD\Surrogate_Wing\rae2822.dat")

# --- 2. Geometry Definition and Airfoil Generation (using NURBS) ---
def generate_airfoil_coordinates_nurbs(design_variables_section, 
                                       baseline_x_full, baseline_y_full,
                                       baseline_x_upper, baseline_y_upper,
                                       baseline_x_lower, baseline_y_lower,
                                       num_points=200):
    """
    Generates airfoil coordinates using three NURBS curves (LE, upper, lower) with C1 continuity.
    Design variables directly perturb the y-coordinates of internal NURBS control points.
    """
    if len(design_variables_section) != NUM_DVS_PER_SECTION:
        raise ValueError(f"Expected {NUM_DVS_PER_SECTION} DVs for section, got {len(design_variables_section)}")

    # 1. Create PchipInterpolators from the provided baseline data
    if len(baseline_x_full) < 2:
        print("Warning: Not enough unique points for full baseline airfoil. Returning flat line.")
        return np.array([0, 0.5, 1, 0.5, 0]), np.array([0, 0.01, 0, -0.01, 0]), {}

    f_baseline_full = PchipInterpolator(baseline_x_full, baseline_y_full)
    
    # Ensure upper/lower baseline data also have enough points
    if len(baseline_x_upper) < 2 or len(baseline_x_lower) < 2:
        print("Warning: Not enough unique points for baseline upper/lower surfaces. Returning flat line.")
        return np.array([0, 0.5, 1, 0.5, 0]), np.array([0, 0.01, 0, -0.01, 0]), {}

    f_baseline_upper_surface = PchipInterpolator(baseline_x_upper, baseline_y_upper)
    f_baseline_lower_surface = PchipInterpolator(baseline_x_lower, baseline_y_lower)

    # --- Extract design variables for each segment ---
    # The design variables now only perturb the internal CPs of the upper and lower surfaces
    dv_idx = 0
    y_upper_internal_cps_perturb = design_variables_section[dv_idx : dv_idx + N_UPPER_INTERNAL_CPS]
    dv_idx += N_UPPER_INTERNAL_CPS

    y_lower_internal_cps_perturb = design_variables_section[dv_idx : dv_idx + N_LOWER_INTERNAL_CPS]
    # dv_idx += N_LOWER_INTERNAL_CPS # Not needed after last group

    # --- Define Control Points for LE Curve (FIXED) ---
    # CP_LE_0: Fixed at (0,0) - the very tip of the leading edge
    # CP_LE_1: Upper point on the LE curve, at X_LE_JUNCTION / 2
    # CP_LE_2: Lower point on the LE curve, at X_LE_JUNCTION / 2
    # CP_LE_3: Junction point, fixed at (X_LE_JUNCTION, 0.0)

    # Determine baseline y-coordinates for fixed LE CPs from the full baseline airfoil
    # We'll make these symmetric for a clean LE, or you could derive them from baseline_y_upper/lower
    y_le_upper_fixed = f_baseline_upper_surface(X_LE_JUNCTION / 2) if X_LE_JUNCTION / 2 >= baseline_x_upper[0] and X_LE_JUNCTION / 2 <= baseline_x_upper[-1] else 0.005
    y_le_lower_fixed = f_baseline_lower_surface(X_LE_JUNCTION / 2) if X_LE_JUNCTION / 2 >= baseline_x_lower[0] and X_LE_JUNCTION / 2 <= baseline_x_lower[-1] else -0.005

    le_control_points = [
        [X_LE_JUNCTION, 0.0, 0.0],                                     # CP_LE_1 (fixed LE tip)
        [X_LE_JUNCTION, y_le_upper_fixed, 0.0],          # CP_LE_2 (upper LE point)
        [X_LE_JUNCTION, y_le_lower_fixed, 0.0],          # CP_LE_3 (lower LE point)
    ]

    # Get CP_LE_2 and CP_LE_3 (junction) for tangency calculation
    cp_le_1 = np.array(le_control_points[0]) # CP_LE_0 is the first point in le_control_points
    cp_le_2 = np.array(le_control_points[1]) # CP_LE_2 is the second point in le_control_points
    cp_le_3 = np.array(le_control_points[2]) # CP_LE_3 is the third point (junction)



    # Baseline y-coords for internal upper/lower CPs
    y_upper_internal_cps_baseline = f_baseline_upper_surface(X_INTERNAL_UPPER_CPS) # Use specific upper interpolator
    y_upper_internal_cps_perturbed = y_upper_internal_cps_baseline + y_upper_internal_cps_perturb

    y_lower_internal_cps_baseline = f_baseline_lower_surface(X_INTERNAL_LOWER_CPS) # Use specific lower interpolator
    y_lower_internal_cps_perturbed = y_lower_internal_cps_baseline + y_lower_internal_cps_perturb

    # Upper Control Points
    upper_control_points = [list(cp_le_1)] # CP_UPPER_0 is CP_LE_3 (junction)
    upper_control_points.append(list(cp_le_2)) # CP_UPPER_1 for tangency
    for x, y_pert in zip(X_INTERNAL_UPPER_CPS, y_upper_internal_cps_perturbed):
        upper_control_points.append([x, y_pert, 0.0])
    # Fixed TE point (1,0)
    upper_control_points.append([1.0, f_baseline_upper_surface(1.0), 0.0]) # CP_UPPER_N (TE)

    # Lower Control Points
    lower_control_points = [list(cp_le_1)] # CP_LOWER_0 is CP_LE_3 (junction)
    lower_control_points.append(list(cp_le_3)) # CP_LOWER_1 for tangency
    for x, y_pert in zip(X_INTERNAL_LOWER_CPS, y_lower_internal_cps_perturbed):
        lower_control_points.append([x, y_pert, 0.0])
    # Fixed TE point (1,0)
    lower_control_points.append([1.0, f_baseline_lower_surface(1.0), 0.0]) # CP_LOWER_N (TE)

    # --- Create NURBS curves ---

    curve_upper = NURBS.Curve()
    curve_upper.degree = NURBS_DEGREE
    curve_upper.ctrlpts = upper_control_points
    curve_upper.knotvector = utilities.generate_knot_vector(curve_upper.degree, len(curve_upper.ctrlpts))

    curve_lower = NURBS.Curve()
    curve_lower.degree = NURBS_DEGREE
    curve_lower.ctrlpts = lower_control_points
    curve_lower.knotvector = utilities.generate_knot_vector(curve_lower.degree, len(curve_lower.ctrlpts))

    # --- Evaluate curves to get airfoil coordinates ---
    # Distribute points proportionally, ensuring at least NURBS_DEGREE + 1 points for each curve
    num_total_points_per_side = num_points // 2 # Total points for upper or lower side

    # Proportion of points for LE segment relative to one side's total points
    prop_main_segment = (1.0 - X_LE_JUNCTION) # Main segment spans from X_LE_JUNCTION to 1.0

    # Ensure at least NURBS_DEGREE + 1 points for each curve, and that the sum is num_total_points_per_side
    num_upper_eval_points = max(NURBS_DEGREE + 1, int(num_total_points_per_side * prop_main_segment))
    num_lower_eval_points = max(NURBS_DEGREE + 1, int(num_total_points_per_side * prop_main_segment))

    # Adjust if the sum is too low or too high
    # This ensures we get roughly num_total_points_per_side points for each side
    # and avoids issues with too few points for NURBS evaluation.
    
    # Recalculate based on actual curve lengths for better distribution
    # (This is a more advanced step, for now, proportional distribution is fine)

    # curve_le.sample_size = num_le_eval_points
    curve_upper.sample_size = num_upper_eval_points
    curve_lower.sample_size = num_lower_eval_points

    # le_eval_points = curve_le.evalpts
    upper_eval_points = curve_upper.evalpts
    lower_eval_points = curve_lower.evalpts

    # Extract x and y coordinates
    x_upper = np.array([pt[0] for pt in upper_eval_points])
    y_upper = np.array([pt[1] for pt in upper_eval_points])
    x_lower = np.array([pt[0] for pt in lower_eval_points])
    y_lower = np.array([pt[1] for pt in lower_eval_points])

    # --- Combine for XFOIL format: TE (upper) -> LE -> TE (lower) ---
    # The goal is to create a single closed loop of points.
    # Order: TE (upper side) -> LE (upper side) -> LE (lower side) -> TE (lower side)

    # 1. Upper surface: from TE to LE (including LE tip)
    # The `upper_eval_points` go from X_LE_JUNCTION to 1.0
    # The `le_eval_points` go from 0.0 to X_LE_JUNCTION
    # We need to combine these, remove duplicates at X_LE_JUNCTION, and reverse.
    
    # Filter upper_eval_points to exclude the first point (which is the junction)
    upper_segment_coords = np.column_stack((x_upper, y_upper))
    upper_segment_coords = upper_segment_coords[upper_segment_coords[:,0] > X_LE_JUNCTION + 1e-6] # Strictly after junction

    # Filter le_eval_points to include the junction and LE tip
    # le_segment_coords = np.column_stack((x_le, y_le))
    # Ensure LE segment goes from LE tip (0,0) to junction (X_LE_JUNCTION, 0)
    # le_segment_coords = le_segment_coords[le_segment_coords[:,0] <= X_LE_JUNCTION + 1e-6]

    # Combine for the full upper contour (from TE to LE)
    # First, concatenate and sort from LE to TE, then reverse for XFOIL format
    full_upper_contour_raw = upper_segment_coords
    full_upper_contour_sorted = full_upper_contour_raw[np.argsort(full_upper_contour_raw[:, 0])]
    _, unique_full_upper_indices = np.unique(full_upper_contour_sorted[:, 0], return_index=True)
    full_upper_contour = full_upper_contour_sorted[unique_full_upper_indices]
    
    x_final_upper = full_upper_contour[::-1, 0] # Reverse to go from TE to LE
    y_final_upper = full_upper_contour[::-1, 1]

    # 2. Lower surface: from LE to TE
    # Filter lower_eval_points to exclude the first point (which is the junction)
    lower_segment_coords = np.column_stack((x_lower, y_lower))
    lower_segment_coords = lower_segment_coords[lower_segment_coords[:,0] > X_LE_JUNCTION + 1e-6] # Strictly after junction

    # Combine for the full lower contour (from LE to TE)
    # The LE segment is already part of the upper contour. We need to start the lower contour
    # from the LE tip (0,0) and go to TE.
    # Since our LE curve is symmetric, we can use the LE segment from the upper contour.
    
    # We need to ensure the lower contour starts at (0,0) and proceeds to TE.
    # The LE curve provides the points from (0,0) to (X_LE_JUNCTION, 0).
    # The lower curve provides points from (X_LE_JUNCTION, 0) to (1,0).
    
    # Filter LE points to exclude the last one (junction) and reverse for lower side
    # le_segment_for_lower = np.column_stack((x_le, y_le))
    # le_segment_for_lower = le_segment_for_lower[le_segment_for_lower[:,0] < X_LE_JUNCTION - 1e-6] # Strictly before junction
    
    # Combine and sort for the full lower contour (from LE to TE)
    full_lower_contour_raw = lower_segment_coords # Reverse LE for lower
    full_lower_contour_sorted = full_lower_contour_raw[np.argsort(full_lower_contour_raw[:, 0])]
    _, unique_full_lower_indices = np.unique(full_lower_contour_sorted[:, 0], return_index=True)
    full_lower_contour = full_lower_contour_sorted[unique_full_lower_indices]

    # Final concatenation: upper (TE to LE) + lower (LE to TE, excluding LE tip duplicate)
    # The LE tip (0,0) is the last point of x_final_upper.
    # The first point of full_lower_contour should be (0,0). We need to exclude it.
    
    # Find the LE tip (0,0) in the full_lower_contour and exclude it for concatenation
    x_final_lower = full_lower_contour[1:, 0] # Exclude first point (0,0)
    y_final_lower = full_lower_contour[1:, 1]

    final_x = np.concatenate((x_final_upper, x_final_lower))
    final_y = np.concatenate((y_final_upper, y_final_lower))

    # Ensure the trailing edge is closed or nearly closed for XFOIL
    if abs(final_y[0] - final_y[-1]) > 1e-4:
        final_y[-1] = final_y[0]

    # Check for NaN or Inf in coordinates
    if np.any(np.isnan(final_x)) or np.any(np.isinf(final_x)) or \
       np.any(np.isnan(final_y)) or np.any(np.isinf(final_y)):
        print("Warning: Generated airfoil coordinates contain NaN or Inf. Returning flat line.")
        return np.array([0, 0.5, 1, 0.5, 0]), np.array([0, 0.01, 0, -0.01, 0]), {}

    # Return the generated airfoil coordinates and the control points for plotting
    all_control_points = {
        'le': le_control_points,
        'upper': upper_control_points,
        'lower': lower_control_points
    }
    return final_x, final_y, all_control_points

# --- 3. Wing Geometry Assembly (Main function to get wing data) ---
def get_wing_geometry(design_variables):
    """
    Generates the 3D wing geometry based on design variables.
    Each section's airfoil shape is determined by a subset of design variables.
    """
    wing_sections = []

    # Calculate spanwise positions for each section
    # For simplicity, distributing sections evenly along the half-span
    span_positions = np.linspace(0, WING_SPAN / 2, NUM_SPAN_SECTIONS)

    for i in range(NUM_SPAN_SECTIONS):
        span_pos = span_positions[i]
        
        # Calculate chord length using linear taper
        chord = CHORD_ROOT - (CHORD_ROOT - CHORD_TIP) * (span_pos / (WING_SPAN / 2))

        # Extract design variables for the current section
        start_idx = i * NUM_DVS_PER_SECTION
        end_idx = start_idx + NUM_DVS_PER_SECTION
        
        # Ensure design_variables has enough elements
        if end_idx > len(design_variables):
            raise ValueError(f"Not enough design variables for section {i}. Expected {NUM_DVS_TOTAL}, got {len(design_variables)}.")

        section_design_variables = design_variables[start_idx:end_idx]

        # Generate airfoil coordinates using the NURBS function
        # This function now returns control points as a dictionary
        x_coords_norm, y_coords_norm, all_cps = generate_airfoil_coordinates_nurbs(
            section_design_variables, 
            RAE2822_X_BASELINE_FULL, RAE2822_Y_BASELINE_FULL,
            RAE2822_X_BASELINE_UPPER, RAE2822_Y_BASELINE_UPPER,
            RAE2822_X_BASELINE_LOWER, RAE2822_Y_BASELINE_LOWER
        )

        # Scale airfoil coordinates by the current section's chord
        x_coords = x_coords_norm * chord
        y_coords = y_coords_norm * chord

        # Scale control points for plotting
        scaled_cps = {}
        for key, cps_list in all_cps.items():
            scaled_cps[key] = [[cp[0] * chord, cp[1] * chord, cp[2]] for cp in cps_list]

        wing_sections.append({
            'span_pos': span_pos,
            'chord': chord,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'control_points': scaled_cps, # Store scaled control points as a dict
            'area': 0.0, # Placeholder, will be calculated in aerodynamic_analysis
            'thickness': np.max(y_coords_norm) - np.min(y_coords_norm) # Normalized thickness
        })
    return wing_sections

# --- Helper for plotting airfoil and control points ---
def plot_airfoil_and_cps_single_section(ax, section, title_suffix="", offset_y=0.0, plot_label=None):
    """Plots a single airfoil section and its NURBS control points."""
    ax.plot(section['x_coords'], section['y_coords'] + offset_y, 'b-', label=plot_label if plot_label else 'Airfoil Shape')

    # Safely check and plot control points
    if 'control_points' in section and section['control_points']:
        cps = section['control_points']
        
        # Plot LE Control Points and connect them
        if 'le' in cps and cps['le']:
            le_cps_x = np.array([cp[0] for cp in cps['le']])
            le_cps_y = np.array([cp[1] for cp in cps['le']])
            ax.plot(le_cps_x, le_cps_y + offset_y, 'ro--', markersize=5, alpha=0.7, label='LE Control Points')
        
        # Plot Upper Control Points and connect them
        if 'upper' in cps and cps['upper']:
            upper_cps_x = np.array([cp[0] for cp in cps['upper']])
            upper_cps_y = np.array([cp[1] for cp in cps['upper']])
            ax.plot(upper_cps_x, upper_cps_y + offset_y, 'go--', markersize=5, alpha=0.7, label='Upper Control Points')
        
        # Plot Lower Control Points and connect them
        if 'lower' in cps and cps['lower']:
            lower_cps_x = np.array([cp[0] for cp in cps['lower']])
            lower_cps_y = np.array([cp[1] for cp in cps['lower']])
            ax.plot(lower_cps_x, lower_cps_y + offset_y, 'mo--', markersize=5, alpha=0.7, label='Lower Control Points')
    else:
        print(f"Warning: 'control_points' not found or empty for section at span {section.get('span_pos', 'N/A')}. Skipping control point plot.")

    ax.set_title(title_suffix)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m) + Span Offset")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    # ax.legend() # Only show legend once for all sections

# --- 4. Live Plotting Function ---
def update_live_plot(design_variables, fig, axes):
    """
    Updates the live plot of the wing geometry and pressure coefficients.
    
    Args:
        design_variables (np.array): Current design variables.
        fig (matplotlib.figure.Figure): The main figure object.
        axes (list): A list containing the main airfoil subplot (axes[0])
                     and a list of Cp subplots (axes[1:]).
                     Specifically, axes[0] is the airfoil plot,
                     and axes[1] to axes[1+NUM_SPAN_SECTIONS-1] are the Cp plots.
    """
    if not ENABLE_LIVE_PLOT:
        return
    if axes is None or len(axes) < (1 + NUM_SPAN_SECTIONS): # 1 for airfoil, NUM_SPAN_SECTIONS for Cp
        print("Warning: axes object is None or not correctly structured in update_live_plot. Skipping plot update.")
        return

    ax_airfoils = axes[0]
    cp_axes = axes[1:] # All other axes are for Cp plots

    # Clear previous plots
    ax_airfoils.clear()
    for ax_cp_single in cp_axes:
        ax_cp_single.clear()

    try:
        wing_data = get_wing_geometry(design_variables)
        
        # Run aerodynamic analysis to get pressure distributions
        # We need to pass the wing_data (which includes the generated airfoil coords)
        # to aerodynamic_analysis to get the Cp data.
        _, _, _, pressure_distributions = aerodynamic_analysis(
            wing_data, FLIGHT_SPEED, ANGLE_OF_ATTACK, AIR_DENSITY
        )

        # Left Subplot: Airfoil Sections with Control Points
        ax_airfoils.set_title("Airfoil Sections along Half-Span (NURBS) with Control Points")
        ax_airfoils.set_xlabel("X (m)")
        ax_airfoils.set_ylabel("Y (m) + Span Offset")
        ax_airfoils.grid(True, linestyle=':', alpha=0.6)
        ax_airfoils.set_aspect('equal', adjustable='box')

        for i, section in enumerate(wing_data):
            offset_y = (section['span_pos'] / (WING_SPAN / 2)) * (CHORD_ROOT / 2) # Scale offset for visibility
            
            # Plot airfoil shape
            ax_airfoils.plot(section['x_coords'], section['y_coords'] + offset_y,
                             label=f"Airfoil Span {section['span_pos']:.1f}m, Chord {section['chord']:.2f}m")
            
            # Plot control points for each section and connect them
            if 'control_points' in section and section['control_points']:
                cps = section['control_points']
                
                # LE Control Points
                if 'le' in cps and cps['le']:
                    le_cps_x = np.array([cp[0] for cp in cps['le']])
                    le_cps_y = np.array([cp[1] for cp in cps['le']])
                    ax_airfoils.plot(le_cps_x, le_cps_y + offset_y, 'ro--', markersize=3, alpha=0.7, linewidth=0.8)
                
                # Upper Control Points
                if 'upper' in cps and cps['upper']:
                    upper_cps_x = np.array([cp[0] for cp in cps['upper']])
                    upper_cps_y = np.array([cp[1] for cp in cps['upper']])
                    ax_airfoils.plot(upper_cps_x, upper_cps_y + offset_y, 'go--', markersize=3, alpha=0.7, linewidth=0.8)
                
                # Lower Control Points
                if 'lower' in cps and cps['lower']:
                    lower_cps_x = np.array([cp[0] for cp in cps['lower']])
                    lower_cps_y = np.array([cp[1] for cp in cps['lower']])
                    ax_airfoils.plot(lower_cps_x, lower_cps_y + offset_y, 'mo--', markersize=3, alpha=0.7, linewidth=0.8)
        
        # Add a single legend for the airfoil types (optional, can get crowded)
        ax_airfoils.legend(loc='lower right', fontsize='small')

        # Right Subplots: Individual Pressure Coefficient Distributions
        for i, pd_data in enumerate(pressure_distributions):
            if i < NUM_SPAN_SECTIONS: # Ensure we don't go out of bounds for cp_axes
                ax_cp_single = cp_axes[i]
                ax_cp_single.set_title(f"Cp Dist. Span {pd_data['span_pos']:.1f}m")
                ax_cp_single.set_xlabel("Normalized Chordwise Position (x/c)")
                ax_cp_single.set_ylabel(r"$C_p$")
                ax_cp_single.grid(True, linestyle=':', alpha=0.6)
                ax_cp_single.invert_yaxis() # Cp plots usually have negative values upwards

                if pd_data['cp_data'] is not None and len(pd_data['cp_data']) > 0:
                    x_over_c = pd_data['cp_data'][:, 0]
                    cp_values = pd_data['cp_data'][:, 2] # Cp values are in the second column

                    # Find the index of the leading edge (minimum x/c) to split upper and lower surfaces
                    le_idx = np.argmin(x_over_c)
                    
                    # Upper surface: from LE to TE (x/c increasing)
                    # XFOIL output typically goes from TE (upper) -> LE -> TE (lower)
                    # So, points from le_idx to the end (TE lower) are the lower surface
                    # And points from le_idx back to the start (TE upper) are the upper surface (reversed)
                    
                    # Upper surface (from LE to TE)
                    x_upper_cp = x_over_c[le_idx::-1] # From LE back to TE (upper)
                    cp_upper = cp_values[le_idx::-1]
                    
                    # Lower surface (from LE to TE)
                    x_lower_cp = x_over_c[le_idx:] # From LE to TE (lower)
                    cp_lower = cp_values[le_idx:]

                    ax_cp_single.plot(x_upper_cp, cp_upper, 'r-', label='Upper Surface Cp')
                    ax_cp_single.plot(x_lower_cp, cp_lower, 'b-', label='Lower Surface Cp')
                ax_cp_single.legend(loc='upper right', fontsize='small') # Optional, might clutter

    except ValueError as e:
        print(f"Error during live plot update: {e}")
        ax_airfoils.text(0.5, 0.5, "Error generating airfoil", transform=ax_airfoils.transAxes,
                     horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)
        for ax_cp_single in cp_axes:
            ax_cp_single.text(0.5, 0.5, "Error generating Cp data", transform=ax_cp_single.transAxes,
                         horizontalalignment='center', verticalalignment='center', color='red', fontsize=12)
    
    fig.tight_layout()
    plt.draw()
    plt.pause(0.01) # Small pause to allow plot to update
# --- 5. Aerodynamic Analysis ---
def run_xfoil_analysis(airfoil_coords, alpha, reynolds, mach_number, temp_id=""): # Added mach_number as an argument
    """
    Runs XFOIL for a given airfoil, angle of attack, Reynolds number, and Mach number.
    Returns CL, CD, and pressure distribution (Cp).
    Requires XFOIL to be installed and accessible in the PATH.
    """
    temp_dat_filename = f"temp_airfoil_{temp_id}.dat"
    polar_filename = f"temp_airfoil_{temp_id}.pol"
    cp_filename = f"temp_airfoil_{temp_id}.cp"

    # Write airfoil data
    with open(temp_dat_filename, "w") as f:
        f.write("Airfoil from MDO\n")
        for x, y in zip(airfoil_coords[0], airfoil_coords[1]):
            f.write(f"{x:.6f} {y:.6f}\n")

    # XFOIL commands
    xfoil_commands = [
        f"LOAD {temp_dat_filename}",
        "PANE",
        "OPER",
        f"VISC {reynolds}",
        f"M {mach_number}", # This is the line to change
        "PACC", # Set up polar accumulation
        polar_filename, # Output polar file name
        "", # Accept default for dump file
        f"ALFA {alpha}",
        "ITER 100", # Increased iterations
        "CPWR", # Write Cp file
        cp_filename, # Output Cp file name
        "QUIT"
    ]

    cl, cd = 0.0, 1.0 # Default dummy values
    cp_data = None

    # 1. Force XFOIL to disable its graphics window internally
    silent_setup = [
        "PLOP",  # Enter Plotting Options menu
        "G F",   # Graphics False (Turn off UI)
        ""       # Empty string simulates pressing 'Enter' to exit PLOP menu
    ]

    # 2. Combine the setup commands with your existing aerodynamic commands
    full_commands = silent_setup + xfoil_commands
    polar_converged = False

    try:
        # Use DEVNULL for stdout and stderr to completely suppress console interaction
        # The 'input' argument provides stdin, so we don't set stdin=devnull
        process = subprocess.run(
            [r"D:\software_resources\XFOIL6.99\xfoil.exe"], # Make sure this path is correct!
            input="\n".join(full_commands),  # Pass the modified commands here
            text=True,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        # This will run silently, but you can still access text via process.stdout if needed

        # Even if stdout/stderr are redirected to devnull, `capture_output=True` will still
        # capture them into process.stdout and process.stderr.
        # This allows  polar_converged = False
        if os.path.exists(polar_filename):
            try:
                with open(polar_filename, "r") as f:
                    lines = f.readlines()
                    # Look for the line containing CL, CD, etc.
                    # Skip header lines (usually start with # or are empty)
                    for line in lines:
                        if not line.strip() or line.startswith("#"):
                            continue
                        try:
                            data = line.split()
                            # A converged polar output line usually has at least 5-6 columns (alpha, CL, CD, Cm, etc.)
                            if len(data) >= 5: 
                                cl = float(data[1])
                                cd = float(data[2])
                                polar_converged = True
                                break
                        except (ValueError, IndexError):
                            continue
            except Exception as e:
                print(f"Error parsing XFOIL polar file {polar_filename}: {e}")
            finally:
                if os.path.exists(polar_filename): os.remove(polar_filename)

        # Check for Cp data if polar converged
        if polar_converged and os.path.exists(cp_filename):
            try:
                # Cp file header is usually 3 lines, so skip_header=3
                temp_cp_data = np.genfromtxt(cp_filename, skip_header=3)
                if temp_cp_data.ndim == 2 and temp_cp_data.shape[0] > 0: # Ensure it's not empty
                    cp_data = temp_cp_data
            except Exception as e:
                print(f"Error parsing XFOIL Cp file {cp_filename}: {e}")
            finally:
                if os.path.exists(cp_filename): os.remove(cp_filename)
        else:
            # If polar didn't converge or Cp file is missing/empty, reset Cp data
            cp_data = None

    except FileNotFoundError:
        print("Error: XFOIL not found. Please ensure it's installed and in your PATH.")
    except Exception as e:
        # This catch-all exception is now less likely to be hit by the stdin/input conflict
        # but could still catch other unexpected issues.
        print(f"An unexpected error occurred while running XFOIL for {temp_id}: {e}")
    finally:
        if os.path.exists(temp_dat_filename): os.remove(temp_dat_filename)

    return cl, cd, cp_data, polar_converged

def aerodynamic_analysis(airfoil_data, flight_speed, angle_of_attack, air_density):
    """
    Performs aerodynamic analysis for the wing.
    Uses XFOIL for 2D sections and integrates for 3D wing properties.
    """
    global last_xfoil_failed_count

    total_drag = 0.0
    total_lift = 0.0
    pressure_distributions = []
    loads_per_span_section = []

    avg_chord = (CHORD_ROOT + CHORD_TIP) / 2
    reynolds_number = (air_density * flight_speed * avg_chord) / (1.81e-5) # Viscosity of air at 15C
    MACH_NUMBER = flight_speed / 343.0 # Approx speed of sound in m/s

    xfoil_failed_count = 0
    q = 0.5 * air_density * flight_speed**2

    for i, section in enumerate(airfoil_data):
        chord = section['chord']
        span_pos = section['span_pos']
        x_airfoil = section['x_coords'] / chord
        y_airfoil = section['y_coords'] / chord

        cl_section, cd_section, cp_data, is_success = run_xfoil_analysis(
            (x_airfoil, y_airfoil), angle_of_attack, reynolds_number, MACH_NUMBER, temp_id=f"sec{i}"
        )

        if cl_section == 0.0 and cd_section == 1.0: # Check for dummy return values
            xfoil_failed_count += 1
            # Assign a very high drag penalty and no lift
            cd_section = 10.0
            cl_section = 0.0

        local_lift_per_unit_span = cl_section * q * chord
        local_drag_per_unit_span = cd_section * q * chord

        # Calculate section width for integration
        if NUM_SPAN_SECTIONS == 1:
            section_width = WING_SPAN / 2
        elif i == 0: # First section, use width to next section
            section_width = (airfoil_data[i+1]['span_pos'] - airfoil_data[i]['span_pos']) / 2
        elif i == NUM_SPAN_SECTIONS - 1: # Last section, use width from previous section
            section_width = (airfoil_data[i]['span_pos'] - airfoil_data[i-1]['span_pos']) / 2
        else: # Middle sections, average of adjacent widths
            section_width = (airfoil_data[i+1]['span_pos'] - airfoil_data[i-1]['span_pos']) / 2

        total_lift += local_lift_per_unit_span * section_width * 2
        total_drag += local_drag_per_unit_span * section_width * 2

        if cp_data is not None:
            pressure_distributions.append({
                'span_pos': span_pos,
                'chord': chord,
                'cp_data': cp_data
            })
            loads_per_span_section.append({
                'span_pos': span_pos,
                'load_per_unit_span': local_lift_per_unit_span if is_success else np.inf # Penalize load if XFOIL failed
            })
        else:
            loads_per_span_section.append({
                'span_pos': span_pos,
                'load_per_unit_span': np.inf
            })

    if xfoil_failed_count > 0:
        print(f"Warning: XFOIL failed for {xfoil_failed_count}/{NUM_SPAN_SECTIONS} sections. Penalizing drag.")

    last_xfoil_failed_count = int(xfoil_failed_count)

    wing_area = 0.0
    if NUM_SPAN_SECTIONS > 1:
        for i in range(NUM_SPAN_SECTIONS - 1):
            chord_i = airfoil_data[i]['chord']
            chord_iplus1 = airfoil_data[i+1]['chord']
            delta_y = airfoil_data[i+1]['span_pos'] - airfoil_data[i]['span_pos']
            wing_area += (chord_i + chord_iplus1) / 2 * delta_y
    else: # Single section, assume rectangular wing for area
        wing_area = airfoil_data[0]['chord'] * WING_SPAN / 2 # Half wing area

    wing_area *= 2 # For full wing

    overall_cd = total_drag / (q * wing_area) if q * wing_area > 0 else 1.0
    overall_cl = total_lift / (q * wing_area) if q * wing_area > 0 else 0.0

    return overall_cd, overall_cl, loads_per_span_section, pressure_distributions
# --- 6. Structural Analysis (Simplified Beam Model) ---
def structural_analysis(loads_per_span_section, airfoil_data, young_modulus, poisson_ratio):
    """
    Performs structural analysis using simplified beam theory.
    Calculates maximum stress and deflection for a cantilever beam.
    Assumes a simplified box-beam cross-section derived from airfoil.
    """
    max_stress = np.inf
    max_deflection = 0.0

    span_points = np.array([sec['span_pos'] for sec in loads_per_span_section])
    loads = np.array([sec['load_per_unit_span'] for sec in loads_per_span_section])

    # If all loads are effectively zero, return 0 stress/deflection
    if np.all(np.abs(loads) < 1e-6): # Check for very small loads
        return 0.0, 0.0

    if len(span_points) < 2: # Handle case with too few sections
        return 1e12, 1e12

    # Create a finer grid for integration, ensuring it covers the full half-span
    fine_span = np.linspace(0, WING_SPAN / 2, 100)

    # Ensure span_points cover the full range [0, WING_SPAN/2] for interpolation
    # Add root (0) and tip (WING_SPAN/2) if not present
    if span_points[0] > 1e-6: # If first point is not at root
        span_points = np.insert(span_points, 0, 0)
        loads = np.insert(loads, 0, loads[0]) # Assume root load is same as first section
    if span_points[-1] < (WING_SPAN / 2 - 1e-6): # If last point is not at tip
        span_points = np.append(span_points, WING_SPAN / 2)
        loads = np.append(loads, 0) # Assume load tapers to zero at tip

    interpolated_loads = np.interp(fine_span, span_points, loads, left=0, right=0) # Extrapolate with 0

    num_fine_points = len(fine_span)
    shear_force = np.zeros(num_fine_points)
    bending_moment = np.zeros(num_fine_points)

    # Integrate from tip (fine_span[-1]) to root (fine_span[0])
    for i in range(num_fine_points - 2, -1, -1):
        dy = fine_span[i+1] - fine_span[i]
        shear_force[i] = shear_force[i+1] + interpolated_loads[i+1] * dy
        bending_moment[i] = bending_moment[i+1] + shear_force[i+1] * dy

    max_moment = np.max(bending_moment)

    # Calculate section properties for the root (most critical for stress)
    if airfoil_data:
        root_chord = airfoil_data[0]['chord']
        root_thickness = airfoil_data[0]['thickness'] # Use 'thickness' key
    else:
        return 1e12, 1e12 # Should not happen if get_wing_geometry works

    # Simplified assumption: beam height is proportional to airfoil thickness, width to chord
    min_dimension = 1e-3 # 1 mm
    beam_height_root = max(min_dimension, root_chord * root_thickness * 0.8) # Use root_thickness
    beam_width_root = max(min_dimension, root_chord * 0.2)

    I_root = (beam_width_root * beam_height_root**3) / 12
    S_root = (beam_width_root * beam_height_root**2) / 6

    if S_root > 1e-12:
        max_stress = max_moment / S_root
    else:
        max_stress = 1e12

    if I_root > 1e-12:
        curvature = bending_moment / (young_modulus * I_root)
    else:
        curvature = np.full_like(bending_moment, 1e12)

    slope = np.zeros(num_fine_points)
    for i in range(1, num_fine_points):
        dy = fine_span[i] - fine_span[i-1]
        slope[i] = slope[i-1] + curvature[i-1] * dy

    deflection = np.zeros(num_fine_points)
    for i in range(1, num_fine_points):
        dy = fine_span[i] - fine_span[i-1]
        deflection[i] = deflection[i-1] + slope[i-1] * dy

    max_deflection = np.max(deflection)

    # Example: Make stress and deflection somewhat dependent on thickness
    # Thicker airfoils might be stronger/stiffer, so lower stress/deflection
    # This is still a dummy, but now uses existing data.
    # Adjusting max_stress and max_deflection based on a simple inverse relationship with thickness
    # Assuming a baseline thickness of, say, 0.1 for scaling
    if root_thickness > 1e-6: # Avoid division by zero
        stress_factor = (0.1 / root_thickness) # If thickness is 0.2, factor is 0.5 (less stress)
        deflection_factor = (root_thickness / 0.1) # If thickness is 0.2, factor is 2.0 (more deflection for same load if not considering I)
        
        # Re-calculating with a simple scaling for demonstration
        # In a real scenario, this would be part of the actual beam theory calculation
        max_stress *= stress_factor
        max_deflection *= deflection_factor

    return max_stress, max_deflection

# --- Live Plotting Global Variables ---
fig_live, axs_live = None, None
iteration_counter_for_plot = 0
function_eval_counter_for_plot = 0
LIVE_PLOT_EVERY_FUNC_EVALS = 1
VISUAL_UPDATE_DX_TOL = 1e-6
_last_callback_x = None
_last_objective_x = None
_last_plotted_x = None
last_xfoil_failed_count = 0
baseline_airfoil_data_for_plot = None
baseline_root_cp_for_plot = None

def _design_changed(x_new, x_old, tol=1e-8):
    if x_old is None:
        return True
    x_new = np.asarray(x_new, dtype=float)
    x_old = np.asarray(x_old, dtype=float)
    return np.linalg.norm(x_new - x_old, ord=np.inf) > tol

def _keep_live_plot_responsive():
    """Process pending GUI events so the live figure does not appear frozen."""
    if not ENABLE_LIVE_PLOT:
        return
    global fig_live
    if fig_live is None:
        return
    try:
        fig_live.canvas.flush_events()
        plt.pause(0.001)
    except Exception:
        # Plot responsiveness should never interrupt optimization.
        pass

# --- 7. Objective Function and Constraints ---
def eval_opt(design_variables):
    global function_eval_counter_for_plot, _last_objective_x, _last_plotted_x, last_xfoil_failed_count

    function_eval_counter_for_plot += 1
    _keep_live_plot_responsive()
    design_variables = np.clip(design_variables, -0.05, 0.05) # Example clipping

    dx_inf = np.nan
    if _last_objective_x is not None:
        dx_inf = float(np.linalg.norm(design_variables - _last_objective_x, ord=np.inf))
    _last_objective_x = design_variables.copy()

    # The get_wing_geometry function now returns the wing_sections list
    airfoil_data = get_wing_geometry(design_variables)

    overall_cd, overall_cl, loads_per_span_section, pressure_distributions = aerodynamic_analysis(
        airfoil_data, FLIGHT_SPEED, ANGLE_OF_ATTACK, AIR_DENSITY
    )

    # Trigger periodic live updates based on objective evaluations.
    should_plot = _design_changed(design_variables, _last_plotted_x, tol=VISUAL_UPDATE_DX_TOL)
    if fig_live is not None and function_eval_counter_for_plot % LIVE_PLOT_EVERY_FUNC_EVALS == 0:
        if should_plot:
            print(f"Objective Func Eval {function_eval_counter_for_plot}: Plotting (dx_inf={dx_inf:.3e})")
            # You might want to run structural analysis here if its results are needed for the plot title
            # or other plot annotations.
            max_stress, max_deflection = structural_analysis(
                loads_per_span_section, airfoil_data, YOUNG_MODULUS, POISSON_RATIO
            )
            
            update_live_plot(
                design_variables,
                fig_live, # Pass the global figure object
                axs_live # Pass the global axes object
            )
            # You can update the main figure title here if needed, e.g.:
            fig_live.suptitle(f"Opt Step {function_eval_counter_for_plot} - L/D: {overall_cl/overall_cd:.4f}, CD: {overall_cd:.4f}, Stress: {max_stress/1e6:.1f} MPa")

            _last_plotted_x = design_variables.copy()
        else:
            print(f"Objective Func Eval {function_eval_counter_for_plot}: Not plotting (dx_inf={dx_inf:.3e} < {VISUAL_UPDATE_DX_TOL})")

    _keep_live_plot_responsive()
    # Return objective and constraints (assuming your optimizer expects this format)
    return [-overall_cl/overall_cd, [constraint_cd(overall_cd), constraint_max_stress(design_variables), constraint_max_deflection(design_variables)]]

def constraint_cd(overall_cd):
    # This is a placeholder for any additional constraints you might want to implement.
    # For example, you could have a constraint on maximum drag coefficient.
    return overall_cd - 0.005 # Example: CD must be less than 0.005

def optimization_callback(xk):
    global iteration_counter_for_plot, _last_callback_x, _last_plotted_x
    iteration_counter_for_plot += 1
    # Avoid title-only redraws when SLSQP callback repeats the same iterate.
    callback_dx_inf = np.nan
    if _last_callback_x is not None:
        callback_dx_inf = float(np.linalg.norm(xk - _last_callback_x, ord=np.inf))

    should_plot_callback = _design_changed(xk, _last_callback_x, tol=VISUAL_UPDATE_DX_TOL) and _design_changed(xk, _last_plotted_x, tol=VISUAL_UPDATE_DX_TOL)
    
    if should_plot_callback:
        print(f"Callback Iteration {iteration_counter_for_plot}: Plotting (dx_inf={callback_dx_inf:.3e})")
        update_live_plot(
            xk,
            fig_live, # Pass the global figure object
            axs_live # Pass the global axes object
        )
        _last_callback_x = np.asarray(xk, dtype=float).copy()
        _last_plotted_x = np.asarray(xk, dtype=float).copy()
    else:
        print(f"Callback Iteration {iteration_counter_for_plot}: Not plotting (dx_inf={callback_dx_inf:.3e} < {VISUAL_UPDATE_DX_TOL})")
        _keep_live_plot_responsive()

def constraint_max_stress(design_variables):
    _keep_live_plot_responsive()
    design_variables = np.clip(design_variables, -0.05, 0.05)
    airfoil_data = get_wing_geometry(design_variables)
    _, _, loads_per_span_section, _ = aerodynamic_analysis(
        airfoil_data, FLIGHT_SPEED, ANGLE_OF_ATTACK, AIR_DENSITY
    )
    max_stress, _ = structural_analysis(
        loads_per_span_section, airfoil_data, YOUNG_MODULUS, POISSON_RATIO
    )
    _keep_live_plot_responsive()
    return  max_stress - MAX_STRESS_ALLOWED

def constraint_max_deflection(design_variables):
    _keep_live_plot_responsive()
    design_variables = np.clip(design_variables, -0.05, 0.05)
    airfoil_data = get_wing_geometry(design_variables)
    _, _, loads_per_span_section, _ = aerodynamic_analysis(
        airfoil_data, FLIGHT_SPEED, ANGLE_OF_ATTACK, AIR_DENSITY
    )
    _, max_deflection = structural_analysis(
        loads_per_span_section, airfoil_data, YOUNG_MODULUS, POISSON_RATIO
    )
    _keep_live_plot_responsive()
    return max_deflection - MAX_DEFLECTION_ALLOWED

# --- 8. Optimization Setup ---
def build_omads_data(args: argparse.Namespace) -> Dict:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if wing_surrogate is None:
        initialize_wing_surrogate(
            csv_path=os.path.abspath(getattr(args, "csv_path", DEFAULT_CSV_PATH)),
            initial_designs=getattr(args, "initial_designs", 100),
            retrain_every_simulated=getattr(args, "retrain_every_simulated", 50),
            ensemble_size=getattr(args, "ensemble_size", 5),
            hidden_size=getattr(args, "hidden_size", 64),
            epochs=getattr(args, "epochs", 250),
            lr=getattr(args, "lr", 1e-3),
            max_predicted_streak=getattr(args, "max_predicted_streak", 4),
            improve_margin=getattr(args, "improve_margin", 1e-4),
            prediction_acceptance_threshold=getattr(args, "prediction_acceptance_threshold", 0.65),
            seed=getattr(args, "seed", 0),
        )

    nvt = NUM_SPAN_SECTIONS * NUM_DVS_PER_SECTION
    baseline = [0.0]*nvt
    param = {
        "name": "Wing-design",
        "baseline": baseline,
        "lb": [-0.05]*nvt,
        "ub": [0.05]*nvt,
        "var_names": [f"d_{i}" for i in range(1, nvt+1)],
        "scaling": 1,
        "post_dir": os.path.join(SCRIPT_DIR, "post"),
        "constraints_type": ["PB", "PB", "PB"]
    }
    options = {
        "seed": args.seed,
        "budget": args.budget,
        "tol": getattr(args, "tol", 1e-3),
        "psize_init": 1.0,
        "display": (not getattr(args, "quiet", False)),
        "opportunistic": False,
        "check_cache": True,
        "store_cache": True,
        "collect_y": False,
        "rich_direction": True,
        "precision": "high",
        "save_results": True,
        "save_coordinates": False,
        "save_all_best": False,
        "parallel_mode": False,
    }

    search = {
            "type": "sampling",
            "s_method": "ACTIVE",
            "ns": 20,
            "visualize": False,
            "criterion": None
            }

    return {"evaluator": {"blackbox": evaluate_wing}, "param": param, "search": search, "options": options}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wing MDO with Bayesian ensemble surrogate assistance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for OMADS and surrogate ensemble.")
    parser.add_argument("--budget", type=int, default=100, help="OMADS evaluation budget.")
    parser.add_argument("--tol", type=float, default=1e-3, help="OMADS stopping tolerance.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="CSV path to store/reuse wing evaluation history.")
    parser.add_argument("--initial-designs", type=int, default=100, help="Direct evaluations before surrogate predictions are allowed.")
    parser.add_argument("--retrain-every-simulated", type=int, default=50, help="Retrain interval measured in new direct evaluations.")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of models in the ensemble.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size for each surrogate model.")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs per ensemble model.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for surrogate training.")
    parser.add_argument("--max-predicted-streak", type=int, default=4, help="Maximum consecutive predicted evaluations.")
    parser.add_argument("--improve-margin", type=float, default=1e-4, help="Objective improvement margin for P(no-improve).")
    parser.add_argument(
        "--prediction-acceptance-level",
        choices=["conservative", "average", "predictive"],
        default="average",
        help=(
            "Prediction acceptance profile using threshold on P(no-improve) OR P(infeasible): "
            "conservative>=0.85, average>=0.65, predictive>=0.50"
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Disable OMADS display output.")
    return parser.parse_args()


def run_optimization(args: Optional[argparse.Namespace] = None):
    global iteration_counter_for_plot, function_eval_counter_for_plot, _last_callback_x, _last_objective_x, _last_plotted_x
    global baseline_airfoil_data_for_plot, baseline_root_cp_for_plot

    if args is None:
        args = argparse.Namespace(
            seed=42,
            budget=100,
            tol=1e-3,
            csv_path=DEFAULT_CSV_PATH,
            initial_designs=100,
            retrain_every_simulated=50,
            ensemble_size=5,
            hidden_size=64,
            epochs=250,
            lr=1e-3,
            max_predicted_streak=4,
            improve_margin=1e-4,
            prediction_acceptance_threshold=0.65,
            quiet=False,
        )

    iteration_counter_for_plot = 0
    function_eval_counter_for_plot = 0
    _last_callback_x = None
    _last_objective_x = None
    _last_plotted_x = None
    baseline_airfoil_data_for_plot = None
    baseline_root_cp_for_plot = None

    num_dvs_total = NUM_SPAN_SECTIONS * NUM_DVS_PER_SECTION
    initial_design_variables = np.zeros(num_dvs_total)
    bounds = [(-0.01, 0.01)] * num_dvs_total

    constraints = [
        {'type': 'ineq', 'fun': constraint_max_stress},
        {'type': 'ineq', 'fun': constraint_max_deflection}
    ]

    print("Starting optimization...")

    # Initial plot before optimization starts
    update_live_plot(initial_design_variables, fig_live, # Pass the global figure object
                axs_live) # Pass the global axes object)
    # time.sleep(1)

    omads_data = build_omads_data(args)
    out, _, _ = mads.main(omads_data)

    optimal_design_variables = out["xmin"]
    optimal_airfoil_data = get_wing_geometry(optimal_design_variables)
    optimal_cd, optimal_cl, optimal_loads, optimal_pressure_distributions = aerodynamic_analysis(
        optimal_airfoil_data, FLIGHT_SPEED, ANGLE_OF_ATTACK, AIR_DENSITY
    )
    optimal_max_stress, optimal_max_deflection = structural_analysis(
        optimal_loads, optimal_airfoil_data, YOUNG_MODULUS, POISSON_RATIO
    )

    print(f"\nVerification of Optimal Design:")
    print(f"  Drag Coefficient: {optimal_cd:.6f}")
    print(f"  Lift Coefficient: {optimal_cl:.6f}")
    print(f"  Max Stress: {optimal_max_stress / 1e6:.2f} MPa (Allowed: {MAX_STRESS_ALLOWED / 1e6:.2f} MPa)")
    print(f"  Max Deflection: {optimal_max_deflection:.4f} m (Allowed: {MAX_DEFLECTION_ALLOWED:.4f} m)")

    update_live_plot(optimal_design_variables, fig_live, axs_live)
    plt.ioff()
    plt.show()

    return out

# for ($i = 1; $i -le 20; $i++) { python CFD\Surrogate_Wing\airfoil_generator_AI.py --budget 400 --initial-designs 100 --retrain-every-simulated 50 --seed $i --csv-path "CFD\Surrogate_Wing\wing_xfoil_AI\wing_XFOIL_eval_log_validation_$i.csv" --quiet }

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    acceptance_threshold_map = {
        "conservative": 0.85,
        "average": 0.65,
        "predictive": 0.50,
    }
    args.prediction_acceptance_threshold = acceptance_threshold_map[args.prediction_acceptance_level]

    if not args.quiet:
        print(
            "Prediction acceptance level: "
            f"{args.prediction_acceptance_level} (threshold={args.prediction_acceptance_threshold:.2f}, "
            "rule: P(no-improve)>=threshold OR P(infeasible)>=threshold)"
        )

    # Initialize global plotting objects *BEFORE* any potential call to eval_opt
    ENABLE_LIVE_PLOT = False
    if ENABLE_LIVE_PLOT:
        fig_live = plt.figure(figsize=(15, 7))
        gs = GridSpec(NUM_SPAN_SECTIONS, 2, figure=fig_live, width_ratios=[1, 1]) # 1:1 width ratio for left/right columns

        # Left subplot for airfoils
        ax_airfoils = fig_live.add_subplot(gs[:, 0]) # Spans all rows in the first column

        # Right subplots for Cp distributions
        cp_axes_list = []
        for i in range(NUM_SPAN_SECTIONS):
            ax_cp_single = fig_live.add_subplot(gs[i, 1]) # Each Cp plot gets a row in the second column
            cp_axes_list.append(ax_cp_single)
        
        # Combine all axes into a single list for `update_live_plot`
        axs_live = [ax_airfoils] + cp_axes_list

    # Create some dummy design variables (e.g., all zeros for baseline)
    initial_design_variables = np.zeros(NUM_DVS_TOTAL) 
    
    # Or some perturbed variables for testing:
    perturbation_magnitude = 0.005 # e.g., +/- 0.5% of chord
    initial_design_variables = (np.random.rand(NUM_DVS_TOTAL) * 2 - 1) * perturbation_magnitude
    if not os.path.exists(r"C:\apps\code\Py_Dev\WS-AIMDO\simple_airfoil\CFD\Surrogate_Wing\rae2822.dat"):
        print("Creating a dummy 'rae2822.dat' file for demonstration purposes.")
        with open(r"C:\apps\code\Py_Dev\WS-AIMDO\simple_airfoil\CFD\Surrogate_Wing\rae2822.dat", "w") as f:
            f.write("RAE2822 (Dummy Data)\n")
            f.write("1.000000 0.000000\n")
            f.write("0.900000 0.010000\n")
            f.write("0.700000 0.030000\n")
            f.write("0.500000 0.045000\n")
            f.write("0.300000 0.050000\n")
            f.write("0.100000 0.035000\n")
            f.write("0.000000 0.000000\n")
            f.write("0.100000 -0.010000\n")
            f.write("0.300000 -0.020000\n")
            f.write("0.500000 -0.025000\n")
            f.write("0.700000 -0.015000\n")
            f.write("0.900000 -0.005000\n")
            f.write("1.000000 0.000000\n")
        print("A simplified 'rae2822.dat' has been created. For better results, replace it with a full dataset.")

    optimization_result = run_optimization(args)