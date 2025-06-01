# ───────────── 탱크 클래스 ────────────────────────────
class Tank:
    def __init__(
        self,
        level,
        tank_num,
        storage,
        runoff_coef_1,
        side_outlet_height_1,
        infiltration_coef,
        runoff_coef_2 = 0,
        side_outlet_height_2 = 0
    ):
        """
        level                                       : 몇번째 탱크인지에 대한 정보
        tank_num                                    : Total number of lank
        storage                                     : 각 탱크의 저류량
        runoff_coefficient, runoff_coefficient_2    : 측면 유출공 계수 (level = 1인 경우에만 runoff_coefficient_2 정의)
        side_outlet_height_1, side_outlet_height_2  : 측면 유출공 높이 (level = 1인 경우에만 side_outlet_height_2 정의, level = 3인 경우 side_outlet_height_1 = 0)
        infiltration Coefficient                    : 바닥 유출공 계수 (level = 3인 경우 infiltration_coefficient = 0)
        """
        self.level = level
        self.tank_num = tank_num
        if level != 1:
            assert runoff_coef_2 == 0 and side_outlet_height_2 == 0
        if level == tank_num:
            assert side_outlet_height_1 == 0
        self.storage = storage
        self.runoff_coef_1 = runoff_coef_1
        self.side_outlet_height_1 = side_outlet_height_1
        self.infiltration_coef = infiltration_coef
        
        self.runoff_coef_2 = runoff_coef_2
        self.side_outlet_height_2 = side_outlet_height_2
    
    def update(self, inflow, Loss=0):
        """
        t 시점에서 그 다음 t+1 시점 상태로 탱크들의 상태 최신화 

        inflow : 강수량 혹은 상위 탱크로부터의 유입량
        Loss   : 실제증발산량, 혹은 상부 탱크에서의 부족량 

        Returns : 저류량 부족 여부, 저류량 부족량, 측면 유출량, 바닥 유출량
        """
        infiltration = 0
        runoff = 0
        self.storage += inflow
        if self.storage < Loss:
            shortage_amount = Loss - self.storage
            self.storage = 0
            return shortage_amount, runoff, infiltration

        if self.level == 1:
            infiltration = self.infiltration_coef * self.storage
            if self.storage > self.side_outlet_height_1: runoff += self.runoff_coef_1 * (self.storage - self.side_outlet_height_1)
            if self.storage > self.side_outlet_height_2: runoff += self.runoff_coef_2 * (self.storage - self.side_outlet_height_2)
        elif self.level != self.tank_num:
            infiltration = self.infiltration_coef * self.storage
            if self.storage > self.side_outlet_height_1: runoff += self.runoff_coef_1 * (self.storage - self.side_outlet_height_1)
        else:
            runoff += self.storage * self.runoff_coef_1
        """
        증발산량이 탱크의 저류량보다 많이 발생하는 경우
        1. 하부 탱크에 부족 저류량 전달
        2. 측면 및 바닥 유출공에서 유출이 발생하지 않음
        """
        
        self.storage -= (Loss + infiltration + runoff)
        shortage_amount = 0
        return shortage_amount, runoff, infiltration

# ───────────── 탱크 모델 ──────────────────────────────
class Tank_model:
    def __init__(self, tanks, area, timesteps):
        """
        tanks : 탱크 모델을 구성하는 탱크 객체
        """
        self.tanks = tanks
        self.area = area
        self.timesteps = timesteps

        self.tank_num = len(tanks)

        self.total_runoff = []

        self.tank_storages = [[] for _ in range(self.tank_num)]

    def update(self, precip, AET):
        """
        precip : 강수량
        AET : 실제 증발산량        
        """
        tank_inflow = precip
        tank_loss = AET
        tank_runoff = 0
        for i in range(self.tank_num):
            shortage_amount, runoff, infiltration = self.tanks[i].update(tank_inflow, tank_loss)
            tank_inflow = infiltration
            tank_loss = shortage_amount
            tank_runoff += runoff
        self.total_runoff.append(tank_runoff * self.area * 1000 / self.timesteps)
        
        for i in range(self.tank_num):
            self.tank_storages[i].append(self.tanks[i].storage)

    def get_history(self):
        return self.total_runoff, self.tank_storages