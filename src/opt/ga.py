import random
import numpy as np
import sys
from typing import Callable, Literal, List

from ..model import Coefs, make_models


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple R² implementation for default objective.
    """
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    return 1 - (ss_res / ss_tot)


class GAOptimizer:
    """
    Genetic Algorithm to optimize tank-model coefficients.

    tank_num         : number of tanks
    area             : watershed area
    timesteps        : model timesteps
    precip           : precipitation time series
    AET              : actual evapotranspiration series
    observed_runoff  : observed runoff series
    objective        : (sim, obs) -> float
    direction        : 'minimize' or 'maximize'
    population_size  : number of chromosomes
    mutation_rate    : probability of mutation
    crossover_rate   : fraction to preserve as elite
    generations      : max number of generations
    early_termination: allowed stagnant generations
    mutation_storage              : mutation scale for storage
    mutation_side_outlet_height   : mutation scale for side outlets
    mutation_runoff_coef          : mutation scale for runoff coefs
    mutation_infiltration_coef    : mutation scale for infiltration coefs
    fixed_storage                 : if not None, list of storage to keep fixed
    fixed_side_outlet_height      : if not None, list to keep fixed
    fixed_runoff_coef             : if not None, list to keep fixed
    fixed_infiltration_coef       : if not None, list to keep fixed
    """

    def __init__(
        self,
        tank_num: int,
        area: float,
        timesteps: float,
        precip: List[float],
        AET: List[float],
        observed_runoff: List[float],
        objective:
            Callable[[np.ndarray, np.ndarray], float] | None = None,
        direction: Literal["minimize", "maximize"] = "minimize",
        population_size: int = 100,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.8,
        generations: int = 200,
        early_termination: int = 1000,
        mutation_storage: float = 4,
        mutation_side_outlet_height: float = 4,
        mutation_runoff_coef: float = 0.01,
        mutation_infiltration_coef: float = 0.01,
        storage_range: list[tuple[float, float]] = None,
        side_outlet_height_range: list[tuple[float, float]] = None,
        runoff_coef_range: list[tuple[float, float]] = None,
        infiltration_coef_range: list[tuple[float, float]] = None,
        fixed_storage: List[float] | None = None,
        fixed_side_outlet_height: List[float] | None = None,
        fixed_runoff_coef: List[float] | None = None,
        fixed_infiltration_coef: List[float] | None = None,
    ) -> None:
        self.tank_num = tank_num
        self.area = area
        self.timesteps = timesteps
        self.precip = precip
        self.AET = AET
        self.observed_runoff = observed_runoff

        # Default objective = R² (maximize)
        if objective is None:
            self.objective = r2_score
            self.up = True
        else:
            self.objective = objective
            if direction == "maximize":
                self.up = True
            elif direction == "minimize":
                self.up = False
            else:
                raise ValueError(
                    "direction must be 'minimize' or 'maximize'"
                )

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.early_termination = early_termination

        self.mutation_storage = mutation_storage
        self.mutation_side_outlet_height = mutation_side_outlet_height
        self.mutation_runoff_coef = mutation_runoff_coef
        self.mutation_infiltration_coef = mutation_infiltration_coef

        self.storage_range = storage_range
        self.side_outlet_height_range = side_outlet_height_range
        self.runoff_coef_range = runoff_coef_range
        self.infiltration_coef_range = infiltration_coef_range

        self.fixed_storage = fixed_storage
        self.fixed_side_outlet_height = fixed_side_outlet_height
        self.fixed_runoff_coef = fixed_runoff_coef
        self.fixed_infiltration_coef = fixed_infiltration_coef

        self.base_kwargs = {
            "storage_range" : storage_range,
            "side_outlet_height_range": side_outlet_height_range,
            "runoff_coef_range": runoff_coef_range,
            "infiltration_coef_range": infiltration_coef_range
        }

        # Initialize population of Coefs
        self.population: List[Coefs] = []
        for _ in range(self.population_size):
            init_kwargs = dict(self.base_kwargs)
            if self.fixed_storage is not None:
                init_kwargs["storage"] = self.fixed_storage
            if self.fixed_side_outlet_height is not None:
                init_kwargs["side_outlet_height"] = (
                    self.fixed_side_outlet_height
                )
            if self.fixed_runoff_coef is not None:
                init_kwargs["runoff_coef"] = self.fixed_runoff_coef
            if self.fixed_infiltration_coef is not None:
                init_kwargs["infiltration_coef"] = (
                    self.fixed_infiltration_coef
                )

            gene = Coefs(self.tank_num, **init_kwargs)
            self.population.append(gene)

        self.history: List[List[tuple[Coefs, float]]] = []

    def score(self, coef: Coefs) -> float:
        """
        Evaluate one chromosome (Coefs) by running the tank model.
        """
        model = make_models(
            self.tank_num, coef, self.area, self.timesteps
        )
        for i in range(len(self.precip)):
            model.update(self.precip[i], self.AET[i])
        
        pred = np.array(model.total_runoff)
        true = np.array(self.observed_runoff)
        return self.objective(pred, true)

    def _safe_mutate(self, value: float, scale: float) -> float:
        """
        Random jump around `value`, clamped to [0.1*scale, 100].
        """
        base = (
            value
            + random.uniform(-1, 1)
            * scale
            * (2 ** random.uniform(0, 4))
        )
        return min(max(0.1 * scale, base), 100.0)

    def _mutate_coef(self, parent: Coefs) -> Coefs:
        """
        Take one Coefs instance and apply mutation to each field,
        then clamp to both Coefs' own ranges and GA table‐specific bounds.
        """
        # 1) Mutate each list
        storage = [
            self._safe_mutate(v, self.mutation_storage)
            for v in parent.storage
        ]
        side_list = [
            self._safe_mutate(v, self.mutation_side_outlet_height)
            for v in parent.side_outlet_height[:-1]
        ] + [0.0]
        runoff_list = [
            self._safe_mutate(v, self.mutation_runoff_coef)
            for v in parent.runoff_coef
        ]
        infil_list = [
            self._safe_mutate(v, self.mutation_infiltration_coef)
            for v in parent.infiltration_coef[:-1]
        ] + [0.0]

        # Build a new Coefs instance
        child = Coefs(
            self.tank_num,
            storage=storage,
            side_outlet_height=side_list,
            runoff_coef=runoff_list,
            infiltration_coef=infil_list,
            **self.base_kwargs
        )

        # 2) Ensure H12 ≤ H11 (top tank outlets)
        if child.side_outlet_height[0] < child.side_outlet_height[1]:
            child.side_outlet_height[1] = (
                child.side_outlet_height[0] * random.uniform(0, 0.8)
            )

        # 3) Clamp to Coefs' generic ranges
        child.clamp()

        # 4) If any field is fixed, overwrite
        if self.fixed_storage is not None:
            child.storage = list(self.fixed_storage)
        if self.fixed_side_outlet_height is not None:
            child.side_outlet_height = list(
                self.fixed_side_outlet_height
            )
        if self.fixed_runoff_coef is not None:
            child.runoff_coef = list(self.fixed_runoff_coef)
        if self.fixed_infiltration_coef is not None:
            child.infiltration_coef = list(
                self.fixed_infiltration_coef
            )

        return child

    def _crossover_coefs(self, a: Coefs, b: Coefs) -> Coefs:
        """
        Arithmetic crossover between two Coefs, then clamp to valid ranges.
        """
        def cross(x: float, y: float) -> float:
            return x + (y - x) * random.random()

        storage = [
            cross(a.storage[i], b.storage[i])
            for i in range(self.tank_num)
        ]
        side_list = [
            cross(a.side_outlet_height[i], b.side_outlet_height[i])
            for i in range(self.tank_num + 1)
        ]
        runoff_list = [
            cross(a.runoff_coef[i], b.runoff_coef[i])
            for i in range(self.tank_num + 1)
        ]
        infil_list = [
            cross(a.infiltration_coef[i], b.infiltration_coef[i])
            for i in range(self.tank_num)
        ] + [0.0]

        child = Coefs(
            tank_num=self.tank_num,
            storage=storage,
            side_outlet_height=side_list,
            runoff_coef=runoff_list,
            infiltration_coef=infil_list,
            **self.base_kwargs
        )

        # Ensure H12 ≤ H11
        if child.side_outlet_height[0] < child.side_outlet_height[1]:
            child.side_outlet_height[1] = (
                child.side_outlet_height[0] * random.uniform(0, 0.8)
            )

        # Clamp to Coefs' generic ranges
        child.clamp()

        # Overwrite fixed fields if any
        if self.fixed_storage is not None:
            child.storage = list(self.fixed_storage)
        if self.fixed_side_outlet_height is not None:
            child.side_outlet_height = list(
                self.fixed_side_outlet_height
            )
        if self.fixed_runoff_coef is not None:
            child.runoff_coef = list(self.fixed_runoff_coef)
        if self.fixed_infiltration_coef is not None:
            child.infiltration_coef = list(
                self.fixed_infiltration_coef
            )

        return child

    def run(self) -> tuple[Coefs, float]:
        """
        GA 알고리즘 수행. 최적 염색체와 성능 반환.
        """
        # 초기 best 값 세팅 (maximize면 -inf, minimize면 +inf)
        if self.up:
            best_score = -np.inf
        else:
            best_score = np.inf

        check_early_count = 0

        for t in range(self.generations):
            # 세대 진행률 출력
            progress = (t + 1) / self.generations * 100

            # 현재 집단 평가
            scored = [
                (gene, self.score(gene))
                for gene in self.population
            ]
            self.history.extend(scored)

            # 엘리트 선택 (상위 (1 - crossover_rate) 비율)
            scored.sort(key=lambda x: x[1], reverse=self.up)
            cutoff = int(self.population_size * (1 - self.crossover_rate))
            elite = scored[:cutoff]
            self.population = [g for g, _ in elite]

            # 현재 최고 성능 업데이트
            top_score = elite[0][1]
            if (self.up and top_score > best_score) or (
                not self.up and top_score < best_score
            ):
                best_score = top_score
                check_early_count = 0
            elif top_score == best_score:
                check_early_count += 1
                if check_early_count == self.early_termination:
                    break

            # 콘솔에 진행률 + best_score 덮어쓰기
            sys.stdout.write(
                f"\r진행률: {progress:.2f}% "
                f"({t + 1}/{self.generations}) | "
                f"best_score: {best_score:.4f}"
            )
            sys.stdout.flush()

            # 새로운 세대 생성: 엘리트 유지 후 자손 추가
            while len(self.population) < self.population_size:
                # 엘리트-염색체에서 2개 염색체 선택 후 교차 
                p1, p2 = random.sample(range(len(elite)), 2)
                parent1 = elite[p1][0]
                parent2 = elite[p2][0]

                # Crossover
                child = self._crossover_coefs(parent1, parent2)

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate_coef(child)

                self.population.append(child)

        # 최종 세대 중 최고 염색체 찾기
        final_scored = [
            (gene, self.score(gene))
            for gene in self.population
        ]
        final_scored.sort(key=lambda x: x[1], reverse=self.up)
        best_gene, best_value = final_scored[0]

        self.history.extend(final_scored)

        print()
        return best_gene, best_value

    def get_history(self) -> list[list[tuple[Coefs, float]]]:
        return self.history
