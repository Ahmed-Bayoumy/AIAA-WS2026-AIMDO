
import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd
from OMADS import mads, poll

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "post", "Sellar_run", "Sellar_eval_log.csv")


@dataclass
class PredictionSummarySellar:
    objective: float
    g1: float
    g2: float
    prob_infeasible: float
    prob_no_improve: float


def _poly_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    cross = [x[i] * x[j] for i in range(len(x)) for j in range(i + 1, len(x))]
    return np.concatenate(([1.0], x, x**2, np.asarray(cross, dtype=np.float64)))


# --- Discipline 1 Function ---
def discipline1(z1, z2, x, y2):
    return z1**2 + z2 + +x -0.2 * y2


# --- Discipline 2 Function ---
def discipline2(z1, z2, y1):
    return np.sqrt(y1) + z1 + z2


# --- Sellar Evaluator Function ---
def sellar_evaluator(x):
    x, z1, z2 = x

    y1_current = 1.0
    y2_current = 1.0

    tolerance = 1e-6
    max_iterations = 100
    iteration = 0

    while iteration < max_iterations:
        y1_prev = y1_current
        y2_prev = y2_current

        y1_current = discipline1(z1, z2, x, y2_prev)
        y2_current = discipline2(z1, z2, y1_current)

        if abs(y1_current - y1_prev) < tolerance and abs(y2_current - y2_prev) < tolerance:
            break
        iteration += 1
    else:
        print(f"Warning: Fixed-point iteration did not converge after {max_iterations} iterations.")

    objective = x**2 + z2 + y1_current + np.exp(-y2_current)
    g1 = 3.16 - y1_current
    g2 = y2_current - 24
    return [float(objective), [float(g1), float(g2)]]


if nn is not None:
    class _RegressorNet(nn.Module):
        def __init__(self, input_dim: int, hidden: int, output_dim: int = 3):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.mean_head = nn.Linear(hidden, output_dim)
            self.logvar_head = nn.Linear(hidden, output_dim)

        def forward(self, x):
            h = self.backbone(x)
            mean = self.mean_head(h)
            logvar = self.logvar_head(h)
            return mean, logvar
else:
    class _RegressorNet:
        pass


class SellarBayesianEnsembleSurrogate:
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
        self.models: List[_RegressorNet] = []
        self.numpy_models: List[Dict[str, np.ndarray]] = []
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None
        self.total_simulated = 0
        self.simulated_since_retrain = 0
        self.predicted_streak = 0
        self.best_feasible_objective = np.inf
        self._torch_available = torch is not None
        self.y_std_floor: Optional[np.ndarray] = None

    def _should_predict(self) -> bool:
        return self.trained and self.predicted_streak < self.max_predicted_streak

    def _prediction_is_acceptable(self, pred: PredictionSummarySellar) -> bool:
        threshold = self.prediction_acceptance_threshold
        if not np.isfinite(pred.prob_infeasible) and not np.isfinite(pred.prob_no_improve):
            return False
        return (np.isfinite(pred.prob_infeasible) and pred.prob_infeasible >= threshold) or (
            np.isfinite(pred.prob_no_improve) and pred.prob_no_improve >= threshold
        )

    def evaluate(self, x: List[float], simulator_fn: Callable[[List[float]], Tuple[float, float, float]]):
        x_np = np.asarray(x, dtype=float)

        if self._should_predict():
            pred = self.predict_with_probabilities(x_np)
            if not np.isfinite(pred.objective) or not np.isfinite(pred.g1) or not np.isfinite(pred.g2):
                # Guardrail: if surrogate emits non-finite values, fall back to a true simulation.
                objective, constraints = simulator_fn(x_np.tolist())
                g1, g2 = constraints
                feasible = g1 <= 0.0 and g2 <= 0.0 and np.isfinite(objective)
                self.predicted_streak = 0
                self.total_simulated += 1
                self.simulated_since_retrain += 1
                self._append_record(x_np, objective, g1, g2, feasible, "simulated", np.nan, np.nan)
                self._maybe_retrain()
                return [float(objective), [float(g1), float(g2)]]
            if not self._prediction_is_acceptable(pred):
                objective, constraints = simulator_fn(x_np.tolist())
                g1, g2 = constraints
                feasible = g1 <= 0.0 and g2 <= 0.0 and np.isfinite(objective)
                self.predicted_streak = 0
                self.total_simulated += 1
                self.simulated_since_retrain += 1
                if feasible and objective < self.best_feasible_objective:
                    self.best_feasible_objective = float(objective)
                self._append_record(
                    x_np,
                    objective,
                    g1,
                    g2,
                    feasible,
                    "simulated",
                    pred.prob_infeasible,
                    pred.prob_no_improve,
                )
                self._maybe_retrain()
                return [float(objective), [float(g1), float(g2)]]
            c1, c2 = pred.g1, pred.g2
            feasible = c1 <= 0.0 and c2 <= 0.0
            self.predicted_streak += 1
            self._append_record(x_np, pred.objective, pred.g1, pred.g2, feasible, "predicted", pred.prob_infeasible, pred.prob_no_improve)
            return [float(pred.objective), [float(c1), float(c2)]]

        objective, constraints = simulator_fn(x_np.tolist())
        g1, g2 = constraints
        feasible = g1 <= 0.0 and g2 <= 0.0 and np.isfinite(objective)
        self.predicted_streak = 0
        self.total_simulated += 1
        self.simulated_since_retrain += 1
        if feasible and objective < self.best_feasible_objective:
            self.best_feasible_objective = float(objective)

        p_infeasible = np.nan
        p_no_improve = np.nan
        if self.trained:
            pred = self.predict_with_probabilities(x_np)
            p_infeasible = pred.prob_infeasible
            p_no_improve = pred.prob_no_improve

        self._append_record(x_np, objective, g1, g2, feasible, "simulated", p_infeasible, p_no_improve)
        self._maybe_retrain()
        return [float(objective), [float(g1), float(g2)]]

    def _append_record(self, x: np.ndarray, objective: float, g1: float, g2: float, feasible: bool, source: str, prob_infeasible: float, prob_no_improve: float):
        row = {
            "iter": len(self.records) + 1,
            "objective_min": float(objective),
            "g1": float(g1),
            "g2": float(g2),
            "feasible": bool(feasible),
            "source": source,
            "prob_infeasible": float(prob_infeasible) if np.isfinite(prob_infeasible) else np.nan,
            "prob_no_improve": float(prob_no_improve) if np.isfinite(prob_no_improve) else np.nan,
            "best_feasible_objective": float(self.best_feasible_objective) if np.isfinite(self.best_feasible_objective) else np.nan,
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
            self.trained = bool(self.models or self.numpy_models)
            self.simulated_since_retrain = 0
            return
        if self.simulated_since_retrain >= self.retrain_every_simulated:
            self._fit_from_simulated()
            self.trained = bool(self.models or self.numpy_models)
            self.simulated_since_retrain = 0

    def _fit_from_simulated(self):
        sim_df = self.df[self.df["source"] == "simulated"].copy()
        sim_df = sim_df.replace([np.inf, -np.inf], np.nan)
        sim_df = sim_df.dropna(subset=["objective_min", "g1", "g2"]).copy()
        if len(sim_df) < max(self.initial_designs, 4):
            return

        x_cols = sorted([c for c in sim_df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
        y_cols = ["objective_min", "g1", "g2"]

        x_data = sim_df[x_cols].to_numpy(dtype=np.float32)
        y_data = sim_df[y_cols].to_numpy(dtype=np.float32)

        finite_mask = np.isfinite(x_data).all(axis=1) & np.isfinite(y_data).all(axis=1)
        x_data = x_data[finite_mask]
        y_data = y_data[finite_mask]
        if len(x_data) < max(self.initial_designs, 4):
            return

        # Keep a minimum predictive spread so probability estimates remain informative.
        self.y_std_floor = np.maximum(np.nanstd(y_data, axis=0) * 0.05, 1e-3)

        self.x_mean = x_data.mean(axis=0)
        self.x_std = x_data.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x_norm = (x_data - self.x_mean) / self.x_std

        self.models = []
        self.numpy_models = []

        if not self._torch_available:
            self._fit_numpy_ensemble(x_norm, y_data)
            return

        x_tensor = torch.from_numpy(x_norm)
        y_tensor = torch.from_numpy(y_data)
        for model_idx in range(self.ensemble_size):
            torch.manual_seed(1000 + model_idx)
            model = _RegressorNet(input_dim=x_tensor.shape[1], hidden=self.hidden_size, output_dim=3)
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            for _ in range(self.epochs):
                mean, logvar = model(x_tensor)
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (logvar + (y_tensor - mean) ** 2 * inv_var)
                loss = nll.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.models.append(model)

    def _fit_numpy_ensemble(self, x_norm: np.ndarray, y_data: np.ndarray):
        features = np.vstack([_poly_features(row) for row in x_norm])
        ridge = 1e-8
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

    def predict_with_probabilities(self, x: np.ndarray) -> PredictionSummarySellar:
        if not self.trained or self.x_mean is None or self.x_std is None:
            raise RuntimeError("Surrogate is not trained yet.")

        x_in = ((x.astype(np.float32) - self.x_mean) / self.x_std).reshape(1, -1)
        means = []
        vars_ = []

        if self.models:
            x_tensor = torch.from_numpy(x_in)
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    mean, logvar = model(x_tensor)
                means.append(mean.cpu().numpy().reshape(-1))
                vars_.append(np.exp(logvar.cpu().numpy().reshape(-1)))
        else:
            features = _poly_features(x_in.reshape(-1))
            for model in self.numpy_models:
                means.append(features @ model["coef"])
                vars_.append(model["variance"])

        means = np.asarray(means)
        vars_ = np.asarray(vars_)
        means = np.where(np.isfinite(means), means, np.nan)
        vars_ = np.where(np.isfinite(vars_), vars_, np.nan)

        if np.isnan(means).all() or np.isnan(vars_).all():
            return PredictionSummarySellar(np.nan, np.nan, np.nan, np.nan, np.nan)

        agg_mean = means.mean(axis=0)
        agg_var = (vars_ + means**2).mean(axis=0) - agg_mean**2
        agg_var = np.where(np.isfinite(agg_var), agg_var, 1e-8)
        agg_var = np.maximum(agg_var, 1e-8)
        agg_std = np.sqrt(agg_var)

        if self.y_std_floor is not None:
            agg_std = np.maximum(agg_std, self.y_std_floor)

        obj_mean, g1_mean, g2_mean = [float(v) for v in agg_mean]
        obj_std, g1_std, g2_std = [float(v) for v in agg_std]

        n_mc = 2000
        obj_samples = self.rng.normal(obj_mean, obj_std, size=n_mc)
        g1_samples = self.rng.normal(g1_mean, g1_std, size=n_mc)
        g2_samples = self.rng.normal(g2_mean, g2_std, size=n_mc)
        prob_infeasible = float(np.mean((g1_samples > 0.0) | (g2_samples > 0.0)))

        if np.isfinite(self.best_feasible_objective):
            prob_no_improve = float(np.mean(obj_samples >= (self.best_feasible_objective - self.improve_margin)))
        else:
            prob_no_improve = np.nan

        # Avoid hard 0/1 collapse from finite-sample Monte Carlo in overconfident regions.
        eps = 1.0 / (n_mc + 2.0)
        prob_infeasible = float(np.clip(prob_infeasible, eps, 1.0 - eps))
        if np.isfinite(prob_no_improve):
            prob_no_improve = float(np.clip(prob_no_improve, eps, 1.0 - eps))

        return PredictionSummarySellar(obj_mean, g1_mean, g2_mean, prob_infeasible, prob_no_improve)


def _bootstrap_sellar_surrogate_from_csv(surrogate: SellarBayesianEnsembleSurrogate):
    if not os.path.exists(surrogate.csv_path):
        return
    try:
        hist = pd.read_csv(surrogate.csv_path)
    except Exception as exc:
        warn(f"Unable to read surrogate history CSV at {surrogate.csv_path}: {exc}")
        return
    if hist.empty:
        return
    required = {"source", "objective_min", "feasible", "g1", "g2"}
    if not required.issubset(set(hist.columns)):
        warn("Existing Sellar surrogate history CSV has incompatible schema; starting fresh in-memory history.")
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
            surrogate.trained = bool(surrogate.models or surrogate.numpy_models)
        except Exception as exc:
            warn(f"Sellar surrogate warm-start training from CSV failed: {exc}")


sellar_surrogate: Optional[SellarBayesianEnsembleSurrogate] = None


def initialize_sellar_surrogate(
    csv_path: str = DEFAULT_CSV_PATH,
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
    global sellar_surrogate

    sellar_surrogate = SellarBayesianEnsembleSurrogate(
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
    _bootstrap_sellar_surrogate_from_csv(sellar_surrogate)


initialize_sellar_surrogate()


def evaluateSellar(x):
    global sellar_surrogate

    if sellar_surrogate is None:
        initialize_sellar_surrogate()

    y = sellar_surrogate.evaluate(x, sellar_evaluator)
    row_idx = sellar_surrogate.df.index[-1]
    last_row = sellar_surrogate.df.iloc[-1]
    pinf = last_row["prob_infeasible"]
    pni = last_row["prob_no_improve"]

    if pd.isna(pinf):
        pinf = 1.0 if (y[1][0] > 0.0 or y[1][1] > 0.0) else 0.0
    if pd.isna(pni):
        sim_df = sellar_surrogate.df[sellar_surrogate.df["source"] == "simulated"]
        if not sim_df.empty:
            feasible_sim = sim_df[sim_df["feasible"] == True]
            ref_obj = float(feasible_sim["objective_min"].min()) if not feasible_sim.empty else float(sim_df["objective_min"].min())
            pni = 1.0 if y[0] >= (ref_obj - sellar_surrogate.improve_margin) else 0.0
        else:
            pni = np.nan

    sellar_surrogate.df.loc[row_idx, "prob_infeasible"] = pinf
    sellar_surrogate.df.loc[row_idx, "prob_no_improve"] = pni
    sellar_surrogate.records[-1]["prob_infeasible"] = float(pinf) if pd.notna(pinf) else np.nan
    sellar_surrogate.records[-1]["prob_no_improve"] = float(pni) if pd.notna(pni) else np.nan
    sellar_surrogate.df.to_csv(sellar_surrogate.csv_path, index=False)

    print(
        f"Sellar eval: source={last_row['source']}, obj={y[0]:.6f}, g1={y[1][0]:.6f}, g2={y[1][1]:.6f}, "
        f"P(infeasible)={pinf if pd.notna(pinf) else float('nan'):.4f}, P(no-improve)={pni if pd.notna(pni) else float('nan'):.4f}"
    )
    return y


def build_problem(seed: int = 10000, budget: int = 400, tol: float = 1e-13, display: bool = True) -> Dict:
    return {
        "evaluator": {"blackbox": evaluateSellar},
        "param": {
            "name": "Sellar",
            "baseline": [1.0, 5.0, 2.0],
            "lb": [0.0, 0.0, 0.0],
            "ub": [10.0, 10.0, 10.0],
            "var_names": ["x", "z1", "z2"],
            "scaling": [1, 1, 1],
            "constraints_type": ["PB", "PB"],
            "mesh_type": "GMESH",
            "post_dir": os.path.join(SCRIPT_DIR, "post"),
            "h_max": np.inf,
            "rho": 1.0,
            "lambda_multipliers": 1,
        },
        "options": {
            "seed": seed,
            "budget": budget,
            "tol": tol,
            "psize_init": 1,
            "display": display,
            "opportunistic": False,
            "check_cache": True,
            "store_cache": True,
            "collect_y": False,
            "rich_direction": True,
            "precision": "low",
            "save_results": False,
            "save_coordinates": False,
            "save_all_best": False,
            "parallel_mode": False,
        },
        "search": {"type": "sampling", "s_method": "ACTIVE", "ns": 5, "visualize": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sellar MDO with Bayesian ensemble surrogate assistance.")
    parser.add_argument("--seed", type=int, default=10000, help="Random seed for OMADS and surrogate ensemble.")
    parser.add_argument("--budget", type=int, default=400, help="OMADS evaluation budget.")
    parser.add_argument("--tol", type=float, default=1e-13, help="OMADS stopping tolerance.")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="CSV path to store/reuse Sellar evaluation history.")
    parser.add_argument("--initial-designs", type=int, default=100, help="Direct evaluations before surrogate predictions are allowed.")
    parser.add_argument("--retrain-every-simulated", type=int, default=50, help="Retrain interval measured in new direct evaluations.")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of models in the ensemble.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size for each neural surrogate.")
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

# for ($i = 1; $i -le 20; $i++) { python CFD/airfoil/Sellar_MDO.py --budget 400 --seed $i --csv-path "CFD/airfoil/post/Sellar_run/Sellar_eval_log_validation_$i.csv" --quiet }
if __name__ == "__main__":
    args = parse_args()
    acceptance_threshold_map = {
        "conservative": 0.85,
        "average": 0.65,
        "predictive": 0.50,
    }
    acceptance_threshold = acceptance_threshold_map[args.prediction_acceptance_level]
    initialize_sellar_surrogate(
        csv_path=os.path.abspath(args.csv_path),
        initial_designs=args.initial_designs,
        retrain_every_simulated=args.retrain_every_simulated,
        ensemble_size=args.ensemble_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        max_predicted_streak=args.max_predicted_streak,
        improve_margin=args.improve_margin,
        prediction_acceptance_threshold=acceptance_threshold,
        seed=args.seed,
    )
    if not args.quiet:
        print(
            "Prediction acceptance level: "
            f"{args.prediction_acceptance_level} (threshold={acceptance_threshold:.2f}, "
            "rule: P(no-improve)>=threshold OR P(infeasible)>=threshold)"
        )
    mads.main(build_problem(seed=args.seed, budget=args.budget, tol=args.tol, display=(not args.quiet)))
