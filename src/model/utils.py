from .tank_model import Tank, Tank_model
from .coefs import Coefs


def make_models(
    tank_num: int,
    coef: Coefs,
    area: float,
    timesteps: float
):
    tanks: list[Tank] = []

    # Top tank (level 1) has two outlets:
    tanks.append(
        Tank(
            level=1,
            tank_num=tank_num,
            storage=coef.storage[0],
            runoff_coef_1=coef.runoff_coef[0],
            side_outlet_height_1=coef.side_outlet_height[0],
            infiltration_coef=coef.infiltration_coef[0],
            runoff_coef_2=coef.runoff_coef[1],
            side_outlet_height_2=coef.side_outlet_height[1],
        )
    )

    # Middle tanks (levels 2…n−1):
    for idx in range(1, tank_num - 1):
        tanks.append(
            Tank(
                level=idx + 1,
                tank_num=tank_num,
                storage=coef.storage[idx],
                runoff_coef_1=coef.runoff_coef[idx + 1],
                side_outlet_height_1=coef.side_outlet_height[idx + 1],
                infiltration_coef=coef.infiltration_coef[idx],
            )
        )

    # Bottom tank (level n) has just one outlet:
    tanks.append(
        Tank(
            level=tank_num,
            tank_num=tank_num,
            storage=coef.storage[-1],
            runoff_coef_1=coef.runoff_coef[tank_num],
            side_outlet_height_1=0,
            infiltration_coef=0,
        )
    )

    return Tank_model(tanks, area, timesteps)