import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from warnings import warn

import numpy as np
import pandas as pd

import OMADS

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


@dataclass
class PredictionSummaryWing:
    objective: float
    cl: float
    sigma: float
    prob_infeasible: float
    prob_no_improve: float


def _poly_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    linear = x
    squared = x ** 2
    cross_terms = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            cross_terms.append(x[i] * x[j])
    if cross_terms:
        return np.concatenate(([1.0], linear, squared, np.asarray(cross_terms, dtype=np.float64)))
    return np.concatenate(([1.0], linear, squared))


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


class BayesianEnsembleWingSurrogate:
    """Deep-ensemble surrogate for the analytic wing design problem.

    Targets are [Cd, Cl, sigma].
    Constraints are interpreted as:
    g1 = |Cl - 1.2| - 0.1 <= 0
    g2 = sigma - 800 <= 0
    """

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
        self.backend = "torch" if self._torch_available else "numpy"

    def evaluate(self, x: List[float], simulator_fn: Callable[[List[float]], Tuple[float, float, float]]):
        x_np = np.asarray(x, dtype=float)

        if self._should_predict():
            pred = self.predict_with_probabilities(x_np)
            if not self._prediction_is_acceptable(pred):
                objective, cl, sigma = simulator_fn(x_np.tolist())
                c1 = abs(cl - 1.2) - 0.1
                c2 = sigma - 800.0
                feasible = c1 <= 0.0 and c2 <= 0.0 and np.isfinite(objective)

                self.predicted_streak = 0
                self.total_simulated += 1
                self.simulated_since_retrain += 1
                if feasible and objective < self.best_feasible_objective:
                    self.best_feasible_objective = float(objective)

                self._append_record(
                    x=x_np,
                    objective=objective,
                    cl=cl,
                    sigma=sigma,
                    c1=c1,
                    c2=c2,
                    feasible=feasible,
                    source="simulated",
                    prob_infeasible=pred.prob_infeasible,
                    prob_no_improve=pred.prob_no_improve,
                )
                self._maybe_retrain()
                return [float(objective), [float(c1), float(c2)]]

            c1 = abs(pred.cl - 1.2) - 0.1
            c2 = pred.sigma - 800.0
            feasible = c1 <= 0.0 and c2 <= 0.0
            self.predicted_streak += 1
            self._append_record(
                x=x_np,
                objective=pred.objective,
                cl=pred.cl,
                sigma=pred.sigma,
                c1=c1,
                c2=c2,
                feasible=feasible,
                source="predicted",
                prob_infeasible=pred.prob_infeasible,
                prob_no_improve=pred.prob_no_improve,
            )
            return [float(pred.objective), [float(c1), float(c2)]]

        objective, cl, sigma = simulator_fn(x_np.tolist())
        c1 = abs(cl - 1.2) - 0.1
        c2 = sigma - 800.0
        feasible = c1 <= 0.0 and c2 <= 0.0 and np.isfinite(objective)

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

        self._append_record(
            x=x_np,
            objective=objective,
            cl=cl,
            sigma=sigma,
            c1=c1,
            c2=c2,
            feasible=feasible,
            source="simulated",
            prob_infeasible=p_infeasible,
            prob_no_improve=p_no_improve,
        )

        self._maybe_retrain()
        return [float(objective), [float(c1), float(c2)]]

    def _should_predict(self) -> bool:
        if not self.trained:
            return False
        if self.predicted_streak >= self.max_predicted_streak:
            return False
        return True

    def _prediction_is_acceptable(self, pred: PredictionSummaryWing) -> bool:
        threshold = self.prediction_acceptance_threshold
        if not np.isfinite(pred.prob_infeasible) and not np.isfinite(pred.prob_no_improve):
            return False
        return (np.isfinite(pred.prob_infeasible) and pred.prob_infeasible >= threshold) or (
            np.isfinite(pred.prob_no_improve) and pred.prob_no_improve >= threshold
        )

    def _append_record(
        self,
        x: np.ndarray,
        objective: float,
        cl: float,
        sigma: float,
        c1: float,
        c2: float,
        feasible: bool,
        source: str,
        prob_infeasible: float,
        prob_no_improve: float,
    ):
        row = {
            "iter": len(self.records) + 1,
            "objective_min": float(objective),
            "cl": float(cl),
            "sigma": float(sigma),
            "c1": float(c1),
            "c2": float(c2),
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
            self.trained = bool(self.models or self.numpy_models)
            self.simulated_since_retrain = 0
            return

        if self.simulated_since_retrain >= self.retrain_every_simulated:
            self._fit_from_simulated()
            self.trained = bool(self.models or self.numpy_models)
            self.simulated_since_retrain = 0

    def _fit_from_simulated(self):
        sim_df = self.df[self.df["source"] == "simulated"].copy()
        if len(sim_df) < max(self.initial_designs, 4):
            return

        x_cols = sorted([c for c in sim_df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
        y_cols = ["objective_min", "cl", "sigma"]

        x_data = sim_df[x_cols].to_numpy(dtype=np.float32)
        y_data = sim_df[y_cols].to_numpy(dtype=np.float32)

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
        feature_matrix = np.vstack([_poly_features(row) for row in x_norm])
        ridge = 1e-8

        for model_idx in range(self.ensemble_size):
            sample_idx = self.rng.integers(0, len(feature_matrix), size=len(feature_matrix))
            x_boot = feature_matrix[sample_idx]
            y_boot = y_data[sample_idx]

            xtx = x_boot.T @ x_boot
            reg = ridge * np.eye(xtx.shape[0])
            coef = np.linalg.solve(xtx + reg, x_boot.T @ y_boot)

            residuals = y_boot - x_boot @ coef
            variance = residuals.var(axis=0, ddof=1) if len(residuals) > 1 else np.full(y_boot.shape[1], 1e-6)
            variance = np.maximum(variance, 1e-6)

            self.numpy_models.append(
                {
                    "coef": coef,
                    "variance": variance,
                }
            )

    def predict_with_probabilities(self, x: np.ndarray) -> PredictionSummaryWing:
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
                mu = mean.cpu().numpy().reshape(-1)
                var = np.exp(logvar.cpu().numpy().reshape(-1))
                means.append(mu)
                vars_.append(var)
        else:
            features = _poly_features(x_in.reshape(-1))
            for model in self.numpy_models:
                mu = features @ model["coef"]
                var = model["variance"]
                means.append(mu)
                vars_.append(var)

        means = np.asarray(means)
        vars_ = np.asarray(vars_)

        agg_mean = means.mean(axis=0)
        agg_var = (vars_ + means ** 2).mean(axis=0) - agg_mean ** 2
        agg_var = np.maximum(agg_var, 1e-8)
        agg_std = np.sqrt(agg_var)

        obj_mean, cl_mean, sigma_mean = [float(v) for v in agg_mean]
        obj_std, cl_std, sigma_std = [float(v) for v in agg_std]

        n_mc = 2000
        obj_samples = self.rng.normal(obj_mean, obj_std, size=n_mc)
        cl_samples = self.rng.normal(cl_mean, cl_std, size=n_mc)
        sigma_samples = self.rng.normal(sigma_mean, sigma_std, size=n_mc)

        infeasible = (np.abs(cl_samples - 1.2) - 0.1 > 0.0) | (sigma_samples - 800.0 > 0.0)
        prob_infeasible = float(np.mean(infeasible))

        if np.isfinite(self.best_feasible_objective):
            no_improve = obj_samples >= (self.best_feasible_objective - self.improve_margin)
            prob_no_improve = float(np.mean(no_improve))
        else:
            prob_no_improve = np.nan

        return PredictionSummaryWing(
            objective=obj_mean,
            cl=cl_mean,
            sigma=sigma_mean,
            prob_infeasible=prob_infeasible,
            prob_no_improve=prob_no_improve,
        )


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, "post", "Wing-design_run", "wing_eval_log.csv")

iteration = 0
wing_surrogate: Optional[BayesianEnsembleWingSurrogate] = None


def wing_model(x: List[float]) -> Tuple[float, float, float]:
    ar, tr, tc = [float(v) for v in x]
    cd = 0.02 + 0.01 / ar + 0.05 * tc ** 2 + 0.01 * (1.0 - tr)
    cl = 0.5 + 0.05 * ar + 0.1 * tr - 0.2 * tc
    sigma = 100.0 / tc + 5.0 * ar + 20.0 * (1.0 - tr)
    return cd, cl, sigma


def _bootstrap_surrogate_from_csv(surrogate: BayesianEnsembleWingSurrogate):
    if not os.path.exists(surrogate.csv_path):
        return

    try:
        hist = pd.read_csv(surrogate.csv_path)
    except Exception as exc:
        warn(f"Unable to read surrogate history CSV at {surrogate.csv_path}: {exc}")
        return

    if hist.empty:
        return

    required = {"source", "objective_min", "feasible", "c1", "c2", "cl", "sigma"}
    if not required.issubset(set(hist.columns)):
        warn("Existing surrogate history CSV has incompatible schema; starting fresh in-memory history.")
        return

    surrogate.df = hist.copy()
    surrogate.records = hist.to_dict(orient="records")

    sim_df = surrogate.df[surrogate.df["source"] == "simulated"].copy()
    surrogate.total_simulated = int(len(sim_df))
    surrogate.simulated_since_retrain = int(len(sim_df) % max(1, surrogate.retrain_every_simulated))
    surrogate.predicted_streak = 0

    feasible_sim = sim_df[sim_df["feasible"] == True]
    if not feasible_sim.empty:
        surrogate.best_feasible_objective = float(feasible_sim["objective_min"].min())
    else:
        surrogate.best_feasible_objective = np.inf

    if torch is None:
        warn("PyTorch is unavailable: using NumPy fallback surrogate backend.")

    if surrogate.total_simulated >= surrogate.initial_designs:
        try:
            surrogate._fit_from_simulated()
            surrogate.trained = bool(surrogate.models or surrogate.numpy_models)
        except Exception as exc:
            warn(f"Surrogate warm-start training from CSV failed: {exc}")
            surrogate.trained = False


def evaluate_wing(x: List[float]):
    global iteration
    global wing_surrogate

    iteration += 1
    y = wing_surrogate.evaluate(x, wing_model)

    row_idx = wing_surrogate.df.index[-1]
    last_row = wing_surrogate.df.iloc[-1]
    source = last_row["source"]
    pinf = last_row["prob_infeasible"]
    pni = last_row["prob_no_improve"]
    cl = float(last_row["cl"])
    sigma = float(last_row["sigma"])

    if pd.isna(pinf):
        pinf = 1.0 if (y[1][0] > 0.0 or y[1][1] > 0.0) else 0.0

    if pd.isna(pni):
        sim_df = wing_surrogate.df[wing_surrogate.df["source"] == "simulated"]
        if not sim_df.empty:
            feasible_sim = sim_df[sim_df["feasible"] == True]
            if not feasible_sim.empty:
                ref_obj = float(feasible_sim["objective_min"].min())
            else:
                ref_obj = float(sim_df["objective_min"].min())
            pni = 1.0 if y[0] >= (ref_obj - wing_surrogate.improve_margin) else 0.0
        else:
            pni = np.nan

    wing_surrogate.df.loc[row_idx, "prob_infeasible"] = pinf
    wing_surrogate.df.loc[row_idx, "prob_no_improve"] = pni
    wing_surrogate.records[-1]["prob_infeasible"] = float(pinf) if pd.notna(pinf) else np.nan
    wing_surrogate.records[-1]["prob_no_improve"] = float(pni) if pd.notna(pni) else np.nan
    wing_surrogate.df.to_csv(wing_surrogate.csv_path, index=False)

    print(
        f"Wing eval {iteration}: source={source}, Cd={y[0]:.6f}, Cl={cl:.6f}, sigma={sigma:.6f}, "
        f"g1={y[1][0]:.6f}, g2={y[1][1]:.6f}, "
        f"P(infeasible)={pinf if pd.notna(pinf) else float('nan'):.4f}, "
        f"P(no-improve)={pni if pd.notna(pni) else float('nan'):.4f}"
    )
    return y


def build_omads_data(args: argparse.Namespace) -> Dict:
    baseline = [12.0, 0.8, 0.15]
    param = {
        "name": "Wing-design",
        "baseline": baseline,
        "lb": [5.0, 0.2, 0.08],
        "ub": [15.0, 0.8, 0.15],
        "var_names": ["AR", "TR", "tc"],
        "scaling": 1,
        "post_dir": os.path.join(SCRIPT_DIR, "post"),
    }
    options = {
        "seed": args.seed,
        "budget": args.budget,
        "tol": 1e-3,
        "psize_init": 1.0,
        "display": True,
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
    parser = argparse.ArgumentParser(description="Wing design optimization with OMADS and a Bayesian ensemble surrogate.")
    parser.add_argument("--budget", type=int, default=400, help="OMADS evaluation budget.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for OMADS and the surrogate ensemble.")
    parser.add_argument("--initial-designs", type=int, default=100, help="Direct evaluations before enabling surrogate predictions.")
    parser.add_argument("--retrain-every-simulated", type=int, default=5, help="Retrain after this many new direct evaluations.")
    parser.add_argument("--ensemble-size", type=int, default=5, help="Number of surrogate models in the ensemble.")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer width for each ensemble member.")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs per ensemble member.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for surrogate training.")
    parser.add_argument("--max-predicted-streak", type=int, default=4, help="Maximum consecutive surrogate-only evaluations.")
    parser.add_argument("--improve-margin", type=float, default=1e-4, help="Minimum objective improvement margin.")
    parser.add_argument(
        "--prediction-acceptance-level",
        choices=["conservative", "average", "predictive"],
        default="predictive",
        help=(
            "Prediction acceptance profile using threshold on P(no-improve) OR P(infeasible): "
            "conservative>=0.85, average>=0.65, predictive>=0.50"
        ),
    )
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH, help="Path for surrogate history CSV.")
    return parser.parse_args()


def main():
    global wing_surrogate

    args = parse_args()
    acceptance_threshold_map = {
        "conservative": 0.85,
        "average": 0.65,
        "predictive": 0.50,
    }
    acceptance_threshold = acceptance_threshold_map[args.prediction_acceptance_level]
    wing_surrogate = BayesianEnsembleWingSurrogate(
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
    if not wing_surrogate._torch_available:
        warn("PyTorch was not found. Falling back to a NumPy ensemble surrogate.")
    _bootstrap_surrogate_from_csv(wing_surrogate)

    data = build_omads_data(args)
    out = OMADS.mads.main(data)
    print(out)


if __name__ == "__main__":
    main()