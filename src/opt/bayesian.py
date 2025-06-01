import optuna
import numpy as np
import os
from typing import Callable, Literal

from ..model import Coefs, make_models


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple R² implementation for default objective.
    """
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    return 1 - (ss_res / ss_tot)


class BayesianOptimizer:
    """
    Bayesian optimization (via Optuna) for tank‐model coefficients.

    tank_num            : number of tanks
    area                : watershed area (e.g. km²)
    timesteps           : total model timesteps
    precip              : global precipitation time series (np.ndarray)
    AET                 : actual evapotranspiration time series (np.ndarray)
    observed_runoff     : observed runoff series (np.ndarray)
    objective           : function (sim, obs) -> float
    direction           : 'minimize' or 'maximize'
    n_trials            : number of Optuna trials
    fixed_storage       : if not None, list of storage to keep fixed
    fixed_side_outlet   : if not None, list of side_outlet heights to fix
    fixed_runoff_coef   : if not None, list of runoff coefs to fix
    fixed_infiltration  : if not None, list of infiltration coefs to fix
    storage_range       : (min, max) storage bounds
    side_outlet_range   : (min, max) side height bounds
    runoff_coef_range   : (min, max) runoff coef bounds
    infiltration_range  : (min, max) infiltration coef bounds
    verbosity           : boolean, or optuna.logging. ...
    """

    def __init__(
        self,
        tank_num: int,
        area: float,
        timesteps: float,
        precip: np.ndarray,
        AET: np.ndarray,
        observed_runoff: np.ndarray,
        objective:
            Callable[[np.ndarray, np.ndarray], float] | None = None,
        direction: Literal["minimize", "maximize"] = "minimize",
        n_trials: int = 100,
        fixed_storage: list[float] | None = None,
        fixed_side_outlet_height: list[float] | None = None,
        fixed_runoff_coef: list[float] | None = None,
        fixed_infiltration_coef: list[float] | None = None,
        storage_range: list[tuple[float, float]] = None,
        side_outlet_height_range: list[tuple[float, float]] = None,
        runoff_coef_range: list[tuple[float, float]] = None,
        infiltration_coef_range: list[tuple[float, float]] = None,
        verbosity=False,
        n_jobs=None
    ) -> None:
        self.set_verbosity(verbosity)
        self.n_jobs = self.set_n_jobs(n_jobs)

        self.tank_num = tank_num
        self.area = area
        self.timesteps = timesteps

        # Convert to numpy arrays for safety
        self.precip = precip
        self.AET = AET
        self.observed_runoff = observed_runoff

        if objective is None:
            self.objective = r2_score
            self.direction = "maximize"
        else:
            if direction not in ("minimize", "maximize"):
                raise ValueError(
                    "direction must be 'minimize' or 'maximize'"
                )
            self.objective = objective
            self.direction = direction

        self.n_trials = n_trials

        # Fixed lists or None
        self.fixed_storage = fixed_storage
        self.fixed_side_outlet_height = fixed_side_outlet_height
        self.fixed_runoff_coef = fixed_runoff_coef
        self.fixed_infiltration_coef = fixed_infiltration_coef

        # Parameter bounds, FIXME using coef class(for None or others)
        temp = Coefs(
            tank_num=tank_num,
            storage_range=storage_range,
            side_outlet_height_range=side_outlet_height_range,
            runoff_coef_range=runoff_coef_range,
            infiltration_coef_range=infiltration_coef_range
        )
        self.storage_range = temp.storage_range
        self.side_outlet_height_range = temp.side_outlet_height_range
        self.runoff_coef_range = temp.runoff_coef_range
        self.infiltration_coef_range = temp.infiltration_coef_range

        # History: list of (params_dict, score)
        self.history: list[tuple[dict, float]] = []

    def _suggest_params(self, trial: optuna.Trial) -> dict:
        """
        Suggest a full set of Coefs parameters for one trial.
        Keyed by:
          s{i}       for storage[i]
          side{i}    for side_outlet_height indices
          run{i}     for runoff_coef[i]
          infil{i}   for infiltration_coef[i]
        """
        params: dict = {}

        # storage_i
        if self.fixed_storage is None:
            for i in range(self.tank_num):
                params[f"s{i}"] = trial.suggest_float(
                    f"s{i}",
                    self.storage_range[i][0],
                    self.storage_range[i][1],
                )

        # side_outlet heights
        if self.fixed_side_outlet_height is None:
            # level1 has two: side0, side1
            params["side0"] = trial.suggest_float(
                "side0",
                self.side_outlet_height_range[0][0],
                self.side_outlet_height_range[0][1],
            )
            params["side1"] = trial.suggest_float(
                "side1",
                self.side_outlet_height_range[0][0],
                params["side0"],
            )
            # levels 2..tank_num-1 → single outlet each
            for i in range(2, self.tank_num):
                params[f"side{i}"] = trial.suggest_float(
                    f"side{i}",
                    self.side_outlet_height_range[i][0],
                    self.side_outlet_height_range[i][1],
                )

        # runoff_coef 0..tank_num
        if self.fixed_runoff_coef is None:
            for i in range(self.tank_num + 1):
                params[f"run{i}"] = trial.suggest_float(
                    f"run{i}",
                    self.runoff_coef_range[i][0],
                    self.runoff_coef_range[i][1],
                )

        # infiltration_coef 0..(tank_num-1), plus last = 0
        if self.fixed_infiltration_coef is None:
            for i in range(self.tank_num - 1):
                params[f"infil{i}"] = trial.suggest_float(
                    f"infil{i}",
                    self.infiltration_coef_range[i][0],
                    self.infiltration_coef_range[i][1],
                )

        params = self._fill_fixed_coefs(params)

        return params

    def _fill_fixed_coefs(self, params: dict) -> Coefs:
        if self.fixed_storage:
            for i in range(len(self.fixed_storage)):
                params[f"s{i}"] = self.fixed_storage[i]

        if self.fixed_side_outlet_height:
            for i in range(len(self.fixed_side_outlet_height)):
                params[f"side{i}"] = self.fixed_side_outlet_height[i]
                
        if self.fixed_runoff_coef:
            for i in range(len(self.fixed_runoff_coef)+1):
                params[f"run{i}"] = self.fixed_storage[i]

        if self.fixed_infiltration_coef:
            for i in range(len(self.fixed_infiltration_coef )):
                params[f"infil{i}"] = self.fixed_infiltration_coef[i]

        # bottom tank outlet = 0
        params[f"side{self.tank_num}"] = 0

        
        params[f"infil{self.tank_num-1}"] = 0.0

        return params

    def _build_coefs(self, params: dict) -> Coefs:
        """
        Given a dict of suggested params, build a Coefs instance.
        """
        # build storage list
        storage = [params[f"s{i}"] for i in range(self.tank_num)]

        # build side_outlet_height list
        side_list: list[float] = []
        side_list.append(params["side0"])
        side_list.append(params["side1"])
        for i in range(2, self.tank_num):
            side_list.append(params[f"side{i}"])
        side_list.append(params[f"side{self.tank_num}"])

        # build runoff_coef list
        runoff_list = [params[f"run{i}"] for i in range(self.tank_num + 1)]

        # build infiltration_coef list
        infil_list = [params[f"infil{i}"] for i in range(self.tank_num - 1)]
        infil_list.append(0.0)

        return Coefs(
            self.tank_num,
            storage=storage,
            side_outlet_height=side_list,
            runoff_coef=runoff_list,
            infiltration_coef=infil_list,
            storage_range=self.storage_range,
            side_outlet_height_range=self.side_outlet_height_range,
            runoff_coef_range=self.runoff_coef_range,
            infiltration_coef_range=self.infiltration_coef_range,
        )

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective: build Coefs, run make_models, compute score.
        """
        params = self._suggest_params(trial)
        coefs = self._build_coefs(params)

        model = make_models(
            self.tank_num, coefs, self.area, self.timesteps
        )
        for i in range(len(self.precip)):
            model.update(self.precip[i], self.AET[i])

        score = self.objective(model.total_runoff, self.observed_runoff)

        # record history
        self.history.append((params, score))
        return score

    def run(self) -> Coefs:
        """
        Run Bayesian optimization. Return best set of Coefs.
        """
        study = optuna.create_study(direction=self.direction)
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            n_jobs=self.n_jobs
        )

        best_trial = study.best_trial
        best_value = study.best_value

        best_coefs = self._fill_fixed_coefs(best_trial.params)
        best_coefs = self._build_coefs(best_trial.params)

        print(f"\nBest Value: {best_trial.value:.4f}")
        return best_coefs, best_value

    def get_history(self) -> list[tuple[dict, float]]:
        """
        Return optimization history as list of (params, score).
        """
        return self.history

    def set_n_jobs(self, n_jobs) -> int:
        if n_jobs is None:
            n_cpus = os.cpu_count()
            n = max(n_cpus - 2, 1)
        else:
            n = n_jobs

        return n 

    def set_verbosity(self, param) -> None:
        """
        Select
            optuna.logging.CRITICAL (aka optuna.logging.FATAL),
            optuna.logging.ERROR,
            optuna.logging.WARNING (aka optuna.logging.WARN),
            optuna.logging.INFO,
            optuna.logging.DEBUG.
        """
        logging_cases = [
            optuna.logging.CRITICAL,
            optuna.logging.FATAL,
            optuna.logging.ERROR,
            optuna.logging.WARNING,
            optuna.logging.WARN,
            optuna.logging.INFO,
            optuna.logging.DEBUG,            
        ]
    
        if param == True:
            verbosity = optuna.logging.INFO
        elif param == False:
            verbosity = optuna.logging.WARNING
        elif param in logging_cases:
            verbosity = param
        else:
            raise KeyError
        
        optuna.logging.set_verbosity(verbosity)