import random
from typing import Sequence, Tuple, Optional

class Coefs:
    def __init__(
        self,
        tank_num: int,
        storage: Optional[list[float]] = None,
        side_outlet_height:
            Optional[list[float]] = None,
        runoff_coef:
            Optional[list[float]] = None,
        infiltration_coef:
            Optional[list[float]] = None,
        storage_range:
            Optional[list[tuple[float, float]]] = None,
        side_outlet_height_range:
            Optional[list[tuple[float, float]]] = None,
        runoff_coef_range:
            Optional[list[tuple[float, float]]] = None,
        infiltration_coef_range:
            Optional[list[tuple[float, float]]] = None,
    ):
        """
        tank_num                  : number of tanks
        storage                   : list of length tank_num (initial storage)
        side_outlet_height        : list of length tank_num + 1
        runoff_coef               : list of length tank_num + 1
        infiltration_coef         : list of length tank_num (last element should be 0)

        storage_range             : list of length tank_num, each a (min, max) tuple
        side_outlet_height_range  : list of length tank_num + 1, each a (min, max) tuple
        runoff_coef_range         : list of length tank_num + 1, each a (min, max) tuple
        infiltration_coef_range   : list of length tank_num, each a (min, max) tuple
        """
        self.tank_num = tank_num

        # If no per‐tank ranges provided, default to a single‐range repeated
        if storage_range is None:
            storage_range = [(0.0, 10.0)] * tank_num
        if side_outlet_height_range is None:
            side_outlet_height_range = [(0.0, 80.0)] * (tank_num + 1)
        if runoff_coef_range is None:
            runoff_coef_range = [(0.01, 0.6)] * (tank_num + 1)
        if infiltration_coef_range is None:
            infiltration_coef_range = [(0.01, 0.6)] * tank_num

        self.tank_num = tank_num

        self.storage_range = storage_range
        self.side_outlet_height_range = side_outlet_height_range
        self.runoff_coef_range = runoff_coef_range
        self.infiltration_coef_range = infiltration_coef_range

        # Storage: either user‐provided or random per tank
        self.storage = (
            storage
            or [self._rand_uniform(bounds) for bounds in storage_range]
        )

        # Side‐outlet heights: either user‐provided or random per index
        self.side_outlet_height = (
            side_outlet_height
            or [
                self._rand_uniform(bounds)
                for bounds in side_outlet_height_range
            ]
        )

        # Runoff coefficients: either user‐provided or random per index
        self.runoff_coef = (
            runoff_coef
            or [
                self._rand_uniform(bounds)
                for bounds in runoff_coef_range
            ]
        )

        # Infiltration coefficients: either user‐provided
        # or random per tank, then append 0
        self.infiltration_coef = (
            infiltration_coef
            or [
                self._rand_uniform(bounds)
                for bounds in infiltration_coef_range
            ] + [0.0]
        )

    def _rand_uniform(self, bounds: Tuple[float, float]) -> float:
        lo, hi = bounds
        return random.uniform(lo, hi)

    def _ordinal(self, n: int) -> str:
        if 10 <= n % 100 <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"

    def clamp(self) -> None:
        """
        Clamp each coefficient list to its stored ranges. Use the
        appropriate per-index range for each coefficient.
        """
        # 1) Clamp storage list
        for i in range(self.tank_num):
            lo, hi = self.storage_range[i]
            if self.storage[i] < lo:
                self.storage[i] = lo
            elif self.storage[i] > hi:
                self.storage[i] = hi

        # 2) Clamp side_outlet_height list (length = tank_num+1)
        for i in range(self.tank_num + 1):
            lo, hi = self.side_outlet_height_range[i]
            val = self.side_outlet_height[i]
            if val < lo:
                self.side_outlet_height[i] = lo
            elif val > hi:
                self.side_outlet_height[i] = hi

        # 3) Clamp runoff_coef list (length = tank_num+1)
        for i in range(self.tank_num + 1):
            lo, hi = self.runoff_coef_range[i]
            val = self.runoff_coef[i]
            if val < lo:
                self.runoff_coef[i] = lo
            elif val > hi:
                self.runoff_coef[i] = hi

        # 4) Clamp infiltration_coef list 
        # (length = tank_num, last is implicitly 0)
        for i in range(self.tank_num):
            lo, hi = self.infiltration_coef_range[i]
            val = self.infiltration_coef[i]
            if val < lo:
                self.infiltration_coef[i] = lo
            elif val > hi:
                self.infiltration_coef[i] = hi

    def get_parameters(self) -> dict:
        """
        Return all coefficients in a flat dictionary, with keys:
            - s0, s1, …, s{tank_num-1}            (storage)
            - side0, side1, …, side{tank_num}      (side outlets)
            - run0, run1, …, run{tank_num}        (runoff coefficients)
            - infil0, infil1, …, infil{tank_num-1} (infiltration coefficients)
        """
        params = {}

        # 1) storage: s0.. s{tank_num-1}
        for i in range(self.tank_num):
            params[f"storage{i}"] = self.storage[i]

        # 2) side_outlet_height: side0.. side{tank_num}
        for i in range(self.tank_num + 1):
            params[f"side{i}"] = self.side_outlet_height[i]

        # 3) runoff_coef: run0.. run{tank_num}
        for i in range(self.tank_num + 1):
            params[f"run{i}"] = self.runoff_coef[i]

        # 4) infiltration_coef: infil0.. infil{tank_num-1}
        for i in range(self.tank_num):
            params[f"infiltration{i}"] = self.infiltration_coef[i]

        return params


    def print(self, path=None):
        lines = []
        def log(line=""):
            print(line)
            lines.append(line)

        log()
        log("─" * 10 + " Top tank " + "─" * 10)
        log(f"        Initial Storage             : {round(self.storage[0], 0)}")
        log(f"        Runoff coefficient 1        : {round(self.runoff_coef[0], 3)}")
        log(f"        Side outlet height 1        : {round(self.side_outlet_height[0], 0)}")
        log(f"        Runoff coefficient 2        : {round(self.runoff_coef[1], 3)}")
        log(f"        Side outlet height 2        : {round(self.side_outlet_height[1], 0)}")
        log(f"        Infiltration Coefficient    : {round(self.infiltration_coef[0], 3)}\n")

        for i in range(1, self.tank_num-1):
            if i == 1:
                label = f"{i + 1}nd tank"
            elif i == 2:
                label = f"{i + 1}rd tank"
            else:
                label = f"{i + 1}th tank"
            log("─" * 10 + f" {label} " + "─" * 10)
            log(f"        Initial Storage             : {round(self.storage[i], 0)}")
            log(f"        Runoff coefficient          : {round(self.runoff_coef[i + 1], 3)}")
            log(f"        Side outlet height          : {round(self.side_outlet_height[i + 1], 0)}")
            log(f"        Infiltration Coefficient    : {round(self.infiltration_coef[i], 3)}\n")

        log("─" * 9 + " Bottom tank " + "─" * 9)
        log(f"        Initial Storage             : {round(self.storage[-1], 0)}")
        log(f"        Runoff coefficient          : {round(self.runoff_coef[-1], 3)}")
        log(f"        Side outlet height          : {round(self.side_outlet_height[-1], 0)}")
        log(f"        Infiltration Coefficient    : {round(self.infiltration_coef[-1], 3)}\n")
        log("─" * 11 + " GA end " + "─" * 10)

        if path:
            # 파일로 저장
            with open(path, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")