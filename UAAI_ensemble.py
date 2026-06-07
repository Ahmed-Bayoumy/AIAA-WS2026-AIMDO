import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


@dataclass
class PredictionSummary:
    objective: float
    csa: float
    abs_cd: float
    prob_infeasible: float
    prob_no_improve: float


@dataclass
class PredictionSummarySingleConstraint:
    objective: float
    metric: float
    prob_infeasible: float
    prob_no_improve: float


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


class BayesianEnsembleSurrogate:
    """Deep-ensemble surrogate with uncertainty for objective and constraint metrics.

    Targets are: [objective_min, csa, abs_cd].
    Constraints are interpreted as c1=0.075-csa<=0 and c2=abs_cd-0.006<=0.
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
        self.rng = np.random.default_rng(seed)

        self.records: List[Dict] = []
        self.df = pd.DataFrame()

        self.trained = False
        self.models: List[_RegressorNet] = []
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None

        self.total_simulated = 0
        self.simulated_since_retrain = 0
        self.predicted_streak = 0
        self.best_feasible_objective = np.inf

        self._torch_available = torch is not None

    def evaluate(self, x: List[float], simulator_fn: Callable[[List[float]], Tuple[float, float, float]]):
        x_np = np.asarray(x, dtype=float)

        if self._should_predict():
            pred = self.predict_with_probabilities(x_np)
            c1 = 0.075 - pred.csa
            c2 = pred.abs_cd - 0.006
            feasible = c1 <= 0.0 and c2 <= 0.0
            self.predicted_streak += 1
            self._append_record(
                x=x_np,
                objective=pred.objective,
                csa=pred.csa,
                abs_cd=pred.abs_cd,
                c1=c1,
                c2=c2,
                feasible=feasible,
                source="predicted",
                prob_infeasible=pred.prob_infeasible,
                prob_no_improve=pred.prob_no_improve,
            )
            return [float(pred.objective), [float(c1), float(c2)]]

        objective, csa, abs_cd = simulator_fn(x_np.tolist())
        c1 = 0.075 - csa
        c2 = abs_cd - 0.006
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
            csa=csa,
            abs_cd=abs_cd,
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
        if not self.trained or not self._torch_available:
            return False
        if self.predicted_streak >= self.max_predicted_streak:
            return False
        return True

    def _append_record(
        self,
        x: np.ndarray,
        objective: float,
        csa: float,
        abs_cd: float,
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
            "csa": float(csa),
            "abs_cd": float(abs_cd),
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
        if not self._torch_available:
            return
        if self.total_simulated < self.initial_designs:
            return

        if not self.trained:
            self._fit_from_simulated()
            self.trained = True
            self.simulated_since_retrain = 0
            return

        if self.simulated_since_retrain >= self.retrain_every_simulated:
            self._fit_from_simulated()
            self.simulated_since_retrain = 0

    def _fit_from_simulated(self):
        sim_df = self.df[self.df["source"] == "simulated"].copy()
        if len(sim_df) < max(self.initial_designs, 4):
            return

        x_cols = sorted([c for c in sim_df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
        y_cols = ["objective_min", "csa", "abs_cd"]

        x_data = sim_df[x_cols].to_numpy(dtype=np.float32)
        y_data = sim_df[y_cols].to_numpy(dtype=np.float32)

        self.x_mean = x_data.mean(axis=0)
        self.x_std = x_data.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x_norm = (x_data - self.x_mean) / self.x_std

        x_tensor = torch.from_numpy(x_norm)
        y_tensor = torch.from_numpy(y_data)

        self.models = []
        for model_idx in range(self.ensemble_size):
            torch.manual_seed(1000 + model_idx)
            model = _RegressorNet(input_dim=x_tensor.shape[1], hidden=self.hidden_size)
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

    def predict_with_probabilities(self, x: np.ndarray) -> PredictionSummary:
        if not self.trained or not self.models or self.x_mean is None or self.x_std is None:
            raise RuntimeError("Surrogate is not trained yet.")

        x_in = ((x.astype(np.float32) - self.x_mean) / self.x_std).reshape(1, -1)
        x_tensor = torch.from_numpy(x_in)

        means = []
        vars_ = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                mean, logvar = model(x_tensor)
            mu = mean.cpu().numpy().reshape(-1)
            var = np.exp(logvar.cpu().numpy().reshape(-1))
            means.append(mu)
            vars_.append(var)

        means = np.asarray(means)
        vars_ = np.asarray(vars_)

        agg_mean = means.mean(axis=0)
        agg_var = (vars_ + means ** 2).mean(axis=0) - agg_mean ** 2
        agg_var = np.maximum(agg_var, 1e-8)
        agg_std = np.sqrt(agg_var)

        obj_mean, csa_mean, abs_cd_mean = [float(v) for v in agg_mean]
        obj_std, csa_std, abs_cd_std = [float(v) for v in agg_std]

        n_mc = 2000
        obj_samples = self.rng.normal(obj_mean, obj_std, size=n_mc)
        csa_samples = self.rng.normal(csa_mean, csa_std, size=n_mc)
        abs_cd_samples = self.rng.normal(abs_cd_mean, abs_cd_std, size=n_mc)

        infeasible = (0.075 - csa_samples > 0.0) | (abs_cd_samples - 0.006 > 0.0)
        prob_infeasible = float(np.mean(infeasible))

        if np.isfinite(self.best_feasible_objective):
            no_improve = obj_samples >= (self.best_feasible_objective - self.improve_margin)
            prob_no_improve = float(np.mean(no_improve))
        else:
            prob_no_improve = np.nan

        return PredictionSummary(
            objective=obj_mean,
            csa=csa_mean,
            abs_cd=abs_cd_mean,
            prob_infeasible=prob_infeasible,
            prob_no_improve=prob_no_improve,
        )


class BayesianEnsembleSurrogateSingleConstraint:
    """Deep-ensemble surrogate for one objective and one inequality metric.

    Targets are: [objective_min, metric].
    Constraint is interpreted as metric - constraint_limit <= 0.
    """

    def __init__(
        self,
        csv_path: str,
        constraint_limit: float,
        metric_name: str,
        initial_designs: int = 20,
        retrain_every_simulated: int = 5,
        ensemble_size: int = 5,
        hidden_size: int = 64,
        epochs: int = 250,
        lr: float = 1e-3,
        max_predicted_streak: int = 4,
        improve_margin: float = 1e-4,
        seed: int = 0,
    ):
        self.csv_path = csv_path
        self.constraint_limit = float(constraint_limit)
        self.metric_name = metric_name
        self.initial_designs = int(initial_designs)
        self.retrain_every_simulated = int(retrain_every_simulated)
        self.ensemble_size = int(ensemble_size)
        self.hidden_size = int(hidden_size)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.max_predicted_streak = int(max_predicted_streak)
        self.improve_margin = float(improve_margin)
        self.rng = np.random.default_rng(seed)

        self.records: List[Dict] = []
        self.df = pd.DataFrame()

        self.trained = False
        self.models: List[_RegressorNet] = []
        self.x_mean: Optional[np.ndarray] = None
        self.x_std: Optional[np.ndarray] = None

        self.total_simulated = 0
        self.simulated_since_retrain = 0
        self.predicted_streak = 0
        self.best_feasible_objective = np.inf

        self._torch_available = torch is not None

    def evaluate(self, x: List[float], simulator_fn: Callable[[List[float]], Tuple[float, float]]):
        x_np = np.asarray(x, dtype=float)

        if self._should_predict():
            pred = self.predict_with_probabilities(x_np)
            c1 = pred.metric - self.constraint_limit
            feasible = c1 <= 0.0
            self.predicted_streak += 1
            self._append_record(
                x=x_np,
                objective=pred.objective,
                metric=pred.metric,
                c1=c1,
                feasible=feasible,
                source="predicted",
                prob_infeasible=pred.prob_infeasible,
                prob_no_improve=pred.prob_no_improve,
            )
            return [float(pred.objective), [float(c1)]]

        objective, metric = simulator_fn(x_np.tolist())
        c1 = metric - self.constraint_limit
        feasible = c1 <= 0.0 and np.isfinite(objective)

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
            metric=metric,
            c1=c1,
            feasible=feasible,
            source="simulated",
            prob_infeasible=p_infeasible,
            prob_no_improve=p_no_improve,
        )

        self._maybe_retrain()
        return [float(objective), [float(c1)]]

    def _should_predict(self) -> bool:
        if not self.trained or not self._torch_available:
            return False
        if self.predicted_streak >= self.max_predicted_streak:
            return False
        return True

    def _append_record(
        self,
        x: np.ndarray,
        objective: float,
        metric: float,
        c1: float,
        feasible: bool,
        source: str,
        prob_infeasible: float,
        prob_no_improve: float,
    ):
        row = {
            "iter": len(self.records) + 1,
            "objective_min": float(objective),
            self.metric_name: float(metric),
            "c1": float(c1),
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
        if not self._torch_available:
            return
        if self.total_simulated < self.initial_designs:
            return

        if not self.trained:
            self._fit_from_simulated()
            self.trained = True
            self.simulated_since_retrain = 0
            return

        if self.simulated_since_retrain >= self.retrain_every_simulated:
            self._fit_from_simulated()
            self.simulated_since_retrain = 0

    def _fit_from_simulated(self):
        sim_df = self.df[self.df["source"] == "simulated"].copy()
        if len(sim_df) < max(self.initial_designs, 4):
            return

        x_cols = sorted([c for c in sim_df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
        y_cols = ["objective_min", self.metric_name]

        x_data = sim_df[x_cols].to_numpy(dtype=np.float32)
        y_data = sim_df[y_cols].to_numpy(dtype=np.float32)

        self.x_mean = x_data.mean(axis=0)
        self.x_std = x_data.std(axis=0)
        self.x_std[self.x_std < 1e-8] = 1.0
        x_norm = (x_data - self.x_mean) / self.x_std

        x_tensor = torch.from_numpy(x_norm)
        y_tensor = torch.from_numpy(y_data)

        self.models = []
        for model_idx in range(self.ensemble_size):
            torch.manual_seed(1000 + model_idx)
            model = _RegressorNet(input_dim=x_tensor.shape[1], hidden=self.hidden_size, output_dim=2)
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

    def predict_with_probabilities(self, x: np.ndarray) -> PredictionSummarySingleConstraint:
        if not self.trained or not self.models or self.x_mean is None or self.x_std is None:
            raise RuntimeError("Surrogate is not trained yet.")

        x_in = ((x.astype(np.float32) - self.x_mean) / self.x_std).reshape(1, -1)
        x_tensor = torch.from_numpy(x_in)

        means = []
        vars_ = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                mean, logvar = model(x_tensor)
            mu = mean.cpu().numpy().reshape(-1)
            var = np.exp(logvar.cpu().numpy().reshape(-1))
            means.append(mu)
            vars_.append(var)

        means = np.asarray(means)
        vars_ = np.asarray(vars_)

        agg_mean = means.mean(axis=0)
        agg_var = (vars_ + means ** 2).mean(axis=0) - agg_mean ** 2
        agg_var = np.maximum(agg_var, 1e-8)
        agg_std = np.sqrt(agg_var)

        obj_mean, metric_mean = [float(v) for v in agg_mean]
        obj_std, metric_std = [float(v) for v in agg_std]

        n_mc = 2000
        obj_samples = self.rng.normal(obj_mean, obj_std, size=n_mc)
        metric_samples = self.rng.normal(metric_mean, metric_std, size=n_mc)

        infeasible = (metric_samples - self.constraint_limit) > 0.0
        prob_infeasible = float(np.mean(infeasible))

        if np.isfinite(self.best_feasible_objective):
            no_improve = obj_samples >= (self.best_feasible_objective - self.improve_margin)
            prob_no_improve = float(np.mean(no_improve))
        else:
            prob_no_improve = np.nan

        return PredictionSummarySingleConstraint(
            objective=obj_mean,
            metric=metric_mean,
            prob_infeasible=prob_infeasible,
            prob_no_improve=prob_no_improve,
        )
