# =============================================================================
#                          Hydrology 2025 (457.207A) 
# Department of Civil and Environmental Engineering, Seoul National University
# Lecturer : Prof. Soohyun YANG
# Teaching Assisstant : Jihwan LIM
# -----------------------------------------------------------------------------
# Introduction : 
# This code provides a practical implementation of the tank model 
# (Sugawara and Maruyama, 1956) to support students in understanding 
# how the theoretical concepts introduced in lecture can be translated 
# into computational procedures. It serves as a supplementary material 
# for the processes-based instruction on the conceptual tank model.
# -----------------------------------------------------------------------------
# Citation :
# If you use or refer to this code in reports or assignments, please cite it as follows (APA 7th edition format):
#
# Lim, J. and Yang, S. (2025). *Tank model implementation in Python (Sugawara & Maruyama, 1956)* (v1.0). 
# [Computer software]. Department of Civil and Environmental Engineering, Seoul National University.
# https://doi.org/10.5281/zenodo.15464005
#
# In-text citation: (Lim and Yang, 2025)
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import MonthLocator, DateFormatter
import numpy as np
import random
import sys
import os
from datetime import timedelta
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

tank_level = 4                                             # TODO-1 탱크 "단" 수 설정 (2단 이상)
                                                           
Area = 601.61                                                 #        유역 면적 설정 (km2) (601.612832489189)
file_path = "./data/3009680_p4.csv"                        # TODO-2 인풋 파일 경로 설정
Timestep = 60 * 60 * 24                                    #  일 단위      시간간격 설정  (유출량의 단위는 cms)
start = 365 * 0                                            # TODO-3 분석 대상 기간 설정   
end   = 365 * 5                                            #  
                                                           
storage_range                   = [0, 50]                  # TODO-4 염색체 초기값 범위 설정                                  
side_outlet_height_range        = [0, 80]                  #                                   
runoff_coefficient_range        = [0.01, 0.6]              #                                       
infiltration_coefficient_range  = [0.01, 0.6]              #                                       
                                                           
mutation_storage                  = 4                      # TODO-5 염색체 변이 정도 설정                          
mutation_side_outlet_height       = 4                      #                           
mutation_runoff_coefficient       = 0.01                   #                               
mutation_infiltration_coefficient = 0.01                   #                               
                                                           
population_size    = 100                                   # TODO-6 GA 알고리즘 인자 설정                                                          
mutation_rate      = 0.3                                   #                                                           
crossover_rate     = 0.8                                   #                                                           
generations        = 200                                   #                                                           
objective_function = "R2"                                  #                                           
early_termination  = 1000                                   #                       
                                                           #
output_path        = "Test1"                               
DO_GA              = True                                 #

try:
    os.mkdir(output_path)
except:
    pass
# ───────────── 한글 폰트 설정 추가 ─────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ───────────── 성능 지표 계산 ─────────────────────────
def calculate_metrics(list1, list2):
    """
    list1, list2 : 비교대상 list 2개
    returns : {평가 지표, 평가 지표에 해당하는 값}
    """

    y_true = np.array(list1)
    y_pred = np.array(list2)
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)

    r2 = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rrmse = rmse / y_mean if y_mean != 0 else np.nan

    return {"R2": round(r2, 4), "RRMSE": round(rrmse, 5)}

# ───────────── 탱크 클래스 ────────────────────────────
class tank:
    def __init__(self, level, storage, runoff_coefficient_1, side_outlet_height_1, infiltration_coefficient, runoff_coefficient_2 = 0, side_outlet_height_2 = 0):
        """
        level                                       : 몇번째 탱크인지에 대한 정보
        storage                                     : 각 탱크의 저류량
        runoff_coefficient, runoff_coefficient_2    : 측면 유출공 계수 (level = 1인 경우에만 runoff_coefficient_2 정의)
        side_outlet_height_1, side_outlet_height_2  : 측면 유출공 높이 (level = 1인 경우에만 side_outlet_height_2 정의, level = 3인 경우 side_outlet_height_1 = 0)
        infiltration Coefficient                    : 바닥 유출공 계수 (level = 3인 경우 infiltration_coefficient = 0)
        """
        self.level = level
        if level != 1:
            assert runoff_coefficient_2 == 0 and side_outlet_height_2 == 0
        if level == tank_level:
            assert side_outlet_height_1 == 0
        self.storage = storage
        self.runoff_coefficient_1 = runoff_coefficient_1
        self.side_outlet_height_1 = side_outlet_height_1
        self.infiltration_coefficient = infiltration_coefficient
        
        self.runoff_coefficient_2 = runoff_coefficient_2
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
            infiltration = self.infiltration_coefficient * self.storage
            if self.storage > self.side_outlet_height_1: runoff += self.runoff_coefficient_1 * (self.storage - self.side_outlet_height_1)
            if self.storage > self.side_outlet_height_2: runoff += self.runoff_coefficient_2 * (self.storage - self.side_outlet_height_2)
        elif self.level != tank_level:
            infiltration = self.infiltration_coefficient * self.storage
            if self.storage > self.side_outlet_height_1: runoff += self.runoff_coefficient_1 * (self.storage - self.side_outlet_height_1)
        else:
            runoff += self.storage * self.runoff_coefficient_1
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
    def __init__(self, tanks):
        """
        tanks : 탱크 모델을 구성하는 탱크 객체
        """
        assert len(tanks) == tank_level
        
        self.tanks = tanks

        self.total_runoff = []

        self.tank_storages = [[] for _ in range(tank_level)]

    def update(self, precip, AET):
        """
        precip : 강수량
        AET : 실제 증발산량        
        """
        tank_inflow = precip
        tank_loss = AET
        tank_runoff = 0
        for i in range(tank_level):
            shortage_amount, runoff, infiltration = self.tanks[i].update(tank_inflow, tank_loss)
            tank_inflow = infiltration
            tank_loss = shortage_amount
            tank_runoff += runoff
        self.total_runoff.append(tank_runoff * Area * 1000 / Timestep)
        
        for i in range(tank_level):
            self.tank_storages[i].append(self.tanks[i].storage)

# ───────────── 파라미터 클래스 ─────────────────────────
class coeffs:
    def __init__(self, rand, storage = [], side_outlet_height = [], runoff_coefficient = [], infiltration_coefficient = []):
        if rand:
            self.storage = [random.uniform(storage_range[0], storage_range[1]) for _ in range(tank_level)]
            
            self.side_outlet_height = [random.uniform(side_outlet_height_range[0], side_outlet_height_range[1])]
            self.side_outlet_height.append(random.uniform(side_outlet_height_range[0], self.side_outlet_height[0]))
            for _ in range(tank_level - 3):
                self.side_outlet_height.append(random.uniform(side_outlet_height_range[0], side_outlet_height_range[1]))
            self.side_outlet_height.append(0)
            
            self.runoff_coefficient = [random.uniform(runoff_coefficient_range[0], runoff_coefficient_range[1]) for _ in range(tank_level + 1)]

            self.infiltration_coefficient = []
            for _ in range(tank_level - 1):
                self.infiltration_coefficient.append(random.uniform(infiltration_coefficient_range[0], infiltration_coefficient_range[1]))
            self.infiltration_coefficient.append(0)
            
        else:
            self.storage = storage
            self.side_outlet_height = side_outlet_height
            self.runoff_coefficient = runoff_coefficient
            self.infiltration_coefficient = infiltration_coefficient

    def print(self, path):
        lines = []
        def log(line=""):
            print(line)
            lines.append(line)

        log()
        log("─" * 10 + " Top tank " + "─" * 10)
        log(f"        Initial Storage             : {round(self.storage[0], 0)}")
        log(f"        Runoff coefficient 1        : {round(self.runoff_coefficient[0], 3)}")
        log(f"        Side outlet height 1        : {round(self.side_outlet_height[0], 0)}")
        log(f"        Runoff coefficient 2        : {round(self.runoff_coefficient[1], 3)}")
        log(f"        Side outlet height 2        : {round(self.side_outlet_height[1], 0)}")
        log(f"        Infiltration Coefficient    : {round(self.infiltration_coefficient[0], 3)}\n")

        for i in range(1, tank_level - 1):
            if i == 1:
                label = f"{i + 1}nd tank"
            elif i == 2:
                label = f"{i + 1}rd tank"
            else:
                label = f"{i + 1}th tank"
            log("─" * 10 + f" {label} " + "─" * 10)
            log(f"        Initial Storage             : {round(self.storage[i], 0)}")
            log(f"        Runoff coefficient          : {round(self.runoff_coefficient[i + 1], 3)}")
            log(f"        Side outlet height          : {round(self.side_outlet_height[i + 1], 0)}")
            log(f"        Infiltration Coefficient    : {round(self.infiltration_coefficient[i], 3)}\n")

        log("─" * 9 + " Bottom tank " + "─" * 9)
        log(f"        Initial Storage             : {round(self.storage[-1], 0)}")
        log(f"        Runoff coefficient          : {round(self.runoff_coefficient[-1], 3)}")
        log(f"        Side outlet height          : {round(self.side_outlet_height[-1], 0)}")
        log(f"        Infiltration Coefficient    : {round(self.infiltration_coefficient[-1], 3)}\n")
        log("─" * 11 + " GA end " + "─" * 10)


        # 파일로 저장
        with open(path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def mutate(self):
        """
        본 코드에서 변이 연산은 Random jump (무작위 도약) 사용 
        """
        def safe_mutate(v, scale): 
            """
            불가능한 값으로의 변이를 막기 위한 함수
            """
            return min(max(0.1 * scale, v + random.uniform(-1, 1) * scale * (2 ** random.uniform(0, 4))), 100)
        storage                  = [safe_mutate(v, mutation_storage) for v in self.storage]                                  
        side_outlet_height       = [safe_mutate(v, mutation_side_outlet_height) for v in self.side_outlet_height[:-1]] + [0]                 
        runoff_coefficient       = [safe_mutate(v, mutation_runoff_coefficient) for v in self.runoff_coefficient]                
        infiltration_coefficient = [safe_mutate(v, mutation_infiltration_coefficient) for v in self.infiltration_coefficient[:-1]] + [0] 
        if side_outlet_height[0] < side_outlet_height[1]:
            side_outlet_height[1] = side_outlet_height[0] * random.uniform(0, 0.8)
        return coeffs(False, storage = storage, side_outlet_height = side_outlet_height, runoff_coefficient = runoff_coefficient, infiltration_coefficient = infiltration_coefficient)
    
    def crossover(self, other):
        """
        본 코드에서 교차 연산은 Arithmetic crossover (산술적 교차) 사용 
        두 염색체 사이의 임의의 내분점을 return함
        """
        def cross(a, b): 
            """
            내분점 계산용 함수
            """
            return a + (b - a) * random.uniform(0, 1)
        storage                  = [cross(v, other.storage[i]) for i, v in enumerate(self.storage)]
        side_outlet_height       = [cross(v, other.side_outlet_height[i]) for i, v in enumerate(self.side_outlet_height)]
        runoff_coefficient       = [cross(v, other.runoff_coefficient[i]) for i, v in enumerate(self.runoff_coefficient)]
        infiltration_coefficient = [cross(v, other.infiltration_coefficient[i]) for i, v in enumerate(self.infiltration_coefficient)]
        if side_outlet_height[0] < side_outlet_height[1]:
            side_outlet_height[1] = side_outlet_height[0] * random.uniform(0, 0.8)
        return coeffs(False, storage = storage, side_outlet_height = side_outlet_height, runoff_coefficient = runoff_coefficient, infiltration_coefficient = infiltration_coefficient)

# ───────────── 모델 생성 함수 ──────────────────────────
def make_models(coeff):
    tanks = []
    tanks.append(tank(1, coeff.storage[0], 
                         coeff.runoff_coefficient[0], coeff.side_outlet_height[0], coeff.infiltration_coefficient[0],
                         coeff.runoff_coefficient[1], coeff.side_outlet_height[1]))
    
    for i in range(1, tank_level - 1):
        tanks.append(tank(i + 1, coeff.storage[i], coeff.runoff_coefficient[i + 1], coeff.side_outlet_height[i + 1], coeff.infiltration_coefficient[i]))    
        
    tanks.append(tank(tank_level, coeff.storage[tank_level - 1], coeff.runoff_coefficient[tank_level], 0, 0))
    
    return Tank_model(tanks)

# ───────────── GA 함수 ────────────────────────────────
"""
본 코드에서 Genetic Algorithm은 다음과 같이 구현됨 S. Katoch et al., 2020 참고

Encoding Scheme : Value
Selection       : Rank selection
Crossover       : Single point crossover
Mutation        : Simple inversion mutation operator
"""
def GA(precip, AET, observed_runoff):
    """
    precip              : 강수량 데이터
    AET                 : 실제증발산량 데이터
    observed_runoff     : 관측된 유량값
    
    returns             : GA 알고리즘 수행 후 도출된 최적 파라미터
    """                   
    if objective_function == "R2":
        up = True
    elif objective_function == "RRMSE":
        up = False 
    
    population = [coeffs(True) for _ in range(population_size)]

    def score(gene):
        """
        gene    : 염색체

        returns : 입력된 염색체 파라미터로 계산된 평가지표 출력
        """
        model = make_models(gene)
        for i in range(len(precip)):
            model.update(precip[i], AET[i])
        return calculate_metrics(model.total_runoff, observed_runoff)[objective_function]
    
    performance_per_generation = []
    
    best_performance  = 1000000000 * (1 + (-2) * int(up))    
    check_early_count = 0

    for t in range(generations):
        progress = (t + 1) / generations * 100
        performance = [(gene, score(gene)) for gene in population]
        performance_scores = [s for _, s in performance]
        performance_per_generation.append(performance_scores)

        # 세대에서 가장 좋은 평가지표를 보유한 엘리트-염색체 선정
        elite = sorted(performance, key=lambda x: x[1], reverse=up)[:int(population_size * (1 - crossover_rate))]
        population = [i[0] for i in elite]

        # 현재 최고 성능 업데이트
        if (up and best_performance < elite[0][1]) or (not up and best_performance > elite[0][1]):
            best_performance = elite[0][1]
            check_early_count = 0
        elif best_performance == elite[0][1]:
            check_early_count += 1
            if check_early_count == early_termination:
                break

        # 한 줄에 진행률 + best_performance 출력 (덮어쓰기)
        sys.stdout.write(f"\r진행률: {progress:.2f}% ({t + 1}/{generations}) | best_performance: {best_performance:.4f}")
        sys.stdout.flush()

        while len(population) < population_size:
            # 엘리트-염색체에서 2개 염색체 선택 후 교차 
            p1, p2 = random.sample(range(len(elite)), 2)
            child = elite[p1][0].crossover(elite[p2][0])

            # 변이확률에 의거하여 변이 발생
            if random.uniform(0, 1) < mutation_rate:
                population.append(child.mutate())
            else:
                population.append(child)

    print()    
    plt.figure(figsize=(12, 6))
    box = plt.boxplot(performance_per_generation, showfliers= False, showmeans=False)

    for median in box['medians']:
        median.set_color('white') 

    
    plt.xlabel("Generation")
    plt.ylabel("Performance")
    xtick_positions = np.linspace(0, generations + 1, 9, dtype=int)

    plt.xticks(xtick_positions, xtick_positions)

    description_text = (
        "※ Boxplot 구성 요소 설명 ※\n"
        "- 상자: Q1 ~ Q3\n"
    )

    y_max = max(max(p) for p in performance_per_generation)
    y_min = min(min(p) for p in performance_per_generation)
    x_max = len(performance_per_generation)

    if objective_function == "R2":
        plt.title(f"세대별 R² 변화 (Box Plot)")
        plt.ylim(0, 1)
        x_loc = x_max - 0.5
        y_loc = y_min + (y_max - y_min) * 0.05
    elif objective_function == "RRMSE":
        plt.title(f"세대별 {objective_function} 변화 (Box Plot)")
        plt.ylim(0,5)
        x_loc = x_max - 0.5
        y_loc = y_max - (y_max - y_min) * 0.05
    else:
        x_loc = x_max - 0.5
        y_loc = y_max * 0.9

    plt.text(x_loc, y_loc, description_text,
             fontsize=10, verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path + "/performance_change")
    return elite[0][0]

# ─────────────── 입력자료 불러오기 ─────────────────────                              

df = pd.read_csv(file_path)

Tot_precip = df['P'].fillna(0).tolist()                       # precip          : 모의기간 강수량 자료 (mm/day)
Tot_AET = df['AET'].fillna(0).tolist()                        # AET             : 모의기간 실제증발산량 자료 (mm/day)
Tot_observed_runoff = df['Q'].fillna(3.68).tolist()           # observed_runoff : 모의기간 유량 자료 (cubic meter)

precip          = Tot_precip[start:end]
AET             = Tot_AET[start:end]
observed_runoff = Tot_observed_runoff[start:end]

df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
df = df.set_index(df.columns[0])

base_date = df.index[0]
start_date = base_date + timedelta(days=start)
end_date = base_date + timedelta(days=end)

Top_tank = tank(1, 1, 0.4, 40, 0.05, 0.001, 30)
middle_tank_1 = tank(2, 50, 0.3, 0.2, 0.1)
middle_tank_2 = tank(3, 60, 0.3, 0.2, 0.1)
Bottom_tank = tank(4, 50, 0.001, 0, 0)

random_TM = Tank_model([Top_tank, middle_tank_1, middle_tank_2, Bottom_tank])

for i in range(len(precip)):
    random_TM.update(precip[i], AET[i])

# 플롯 데이터 설정
dates = pd.date_range(start=start_date, periods=len(random_TM.total_runoff), freq="D")

plt.figure(figsize=(14, 6))
if end - start > 365:
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
else:
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.plot(dates, random_TM.total_runoff, label="Random Params", linewidth = 0.7, color='blue')
plt.plot(dates, observed_runoff, label="Observed", linewidth = 1, color='orange')
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Runoff [m³/s]", fontsize = 18)
plt.title("Random Parameter Simulation vs Observed Runoff", fontsize = 22)
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
if end - start > 365:
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
else:
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
for i in range(tank_level):
    plt.plot(dates, random_TM.tank_storages[i], label = f"{i + 1} th tank")
plt.legend()
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Runoff [mm]", fontsize = 18)
plt.title("Storage for each tanks", fontsize = 22)
plt.show()


# ───────────── GA-algorithm  ──────────
if DO_GA:
    best_param = GA(precip             = precip,
                    AET                = AET,
                    observed_runoff    = observed_runoff)

    best_param.print(output_path + "/tank_parameters.txt")

    best_model = make_models(best_param)
    for i in range(len(precip)):
        best_model.update(precip[i], AET[i])

    # 기존에 정의한 start_date 기준으로 날짜 생성
    dates = pd.date_range(start=start_date, periods=len(best_model.total_runoff), freq='D')

    # 결과 저장용 DataFrame 생성
    df_result = pd.DataFrame({
        "Date": dates,
        "Simulated Runoff": best_model.total_runoff
    })
    file_path = os.path.join(output_path, "simulated_runof.csv")
    df_result.to_csv(file_path, index=False, encoding='utf-8-sig')

    # 최대값 계산
    runoff_max = max(max(best_model.total_runoff), max(observed_runoff)) * 2 
    precip_max = max(precip) * 2.5

    fig, ax1 = plt.subplots(figsize=(20, 8))

    # 시뮬레이션 및 관측 유출량 라인 플롯
    ax1.plot(dates, best_model.total_runoff, label="Simulated Runoff", color='black')
    ax1.plot(dates, observed_runoff, label="Observed Runoff", color='tab:orange')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Runoff [m³/s]")
    ax1.set_ylim(0, runoff_max)
    ax1.legend(loc='upper left')
    ax1.set_title("Best Runoff Simulation")

    # x축 월 단위 설정 (1개월 or 3개월)
    if end - start > 365:
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=45)

    # 강수량 바 차트
    ax2 = ax1.twinx()
    ax2.bar(dates, precip, label="Precipitation", color='tab:blue', width=1.0)
    ax2.set_ylabel("Precipitation [mm]")
    ax2.set_ylim(precip_max, 0)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path + "/simulated_runoff_hydrograph")
    plt.show()