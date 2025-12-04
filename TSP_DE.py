# encoding: utf-8
import random
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy
matplotlib.rc('font', family='Microsoft YaHei')

x_init = None
score_init = 1e-9

class Individual(object):
    """个体类"""
    def __init__(self, position=None):
        self.position = np.array(position)
        self.rank = np.argsort(position)
        self.fitness = score_init

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 2) 读取城市坐标
CITY_FILE = "D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\distanceMatrix.txt"
cities = []
with open(CITY_FILE, "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, x, y = line.split("\t")
        cities.append((float(x), float(y), name))

# 3) 定义超参数
POPULATION_SIZE = 120
MAX_GENERATIONS = 2000
CROSSOVER_RATE = 0.9
SCALING_FACTOR = 0.6
N_CITIES = len(cities)

# 4) 预计算距离矩阵以提高效率
def compute_distance_matrix():
    """预计算所有城市间的距离矩阵"""
    coordinates = np.array([(x, y) for x, y, _ in cities])
    dx = coordinates[:, 0][:, None] - coordinates[:, 0]
    dy = coordinates[:, 1][:, None] - coordinates[:, 1]
    return np.sqrt(dx**2 + dy**2)

distance_matrix = compute_distance_matrix()

# 5) 初始化种群
def create_individual():
    """创建单个个体"""
    return Individual(np.random.rand(N_CITIES))

population = [create_individual() for _ in range(POPULATION_SIZE)]

# 6) 定义评估函数
def evaluate_fitness(individual):
    """计算路径长度并返回适应度（路径长度的倒数）"""
    route = individual.rank
    # 计算回路总距离
    indices = np.append(route, route[0])
    path_distances = distance_matrix[indices[:-1], indices[1:]]
    total_distance = np.sum(path_distances)
    return 1.0 / max(1e-12, total_distance)

# 7) 差分进化主循环
convergence_history = []
best_fitness = 0.0
best_solution = None

for generation in range(MAX_GENERATIONS):
    # 评估当前种群
    for individual in population:
        individual.fitness = evaluate_fitness(individual)
        
        # 更新全局最优解
        if individual.fitness > best_fitness:
            best_fitness = individual.fitness
            best_solution = individual.rank.copy()
    
    # 记录当前最优距离
    convergence_history.append(1.0 / best_fitness)
    
    # 创建新一代种群
    new_population = []
    
    for target_idx, target in enumerate(population):
        # 选择三个不同的随机个体
        candidate_indices = [i for i in range(POPULATION_SIZE) if i != target_idx]
        r0, r1, r2 = np.random.choice(candidate_indices, 3, replace=False)
        
        # 差分变异：v = x_r0 + F * (x_r1 - x_r2)
        base_vector = population[r0].position
        donor_vector = base_vector + SCALING_FACTOR * (
            population[r1].position - population[r2].position
        )
        
        # 二项式交叉
        trial_vector = target.position.copy()
        crossover_mask = np.random.rand(N_CITIES) < CROSSOVER_RATE
        
        # 确保至少有一个维度发生交叉
        j_rand = np.random.randint(0, N_CITIES)
        crossover_mask[j_rand] = True
        
        # 应用交叉
        trial_vector[crossover_mask] = donor_vector[crossover_mask]
        
        # 创建试验个体并评估
        trial_individual = Individual(trial_vector)
        trial_fitness = evaluate_fitness(trial_individual)
        
        # 贪婪选择
        if trial_fitness > target.fitness:
            new_population.append(trial_individual)
        else:
            new_population.append(Individual(target.position.copy()))
    
    # 更新种群
    population = new_population
    
    # 定期输出进度
    if generation % 500 == 0:
        current_best_distance = 1.0 / best_fitness
        print(f"迭代 {generation:4d} 代，当前最优距离: {current_best_distance:.6f}")

# 8) 输出结果
final_distance = 1.0 / best_fitness
print(f"\n经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_distance:.6f}")
print("遍历城市顺序为：")
for idx in best_solution:
    print(cities[idx][2], end=' -> ')
print(cities[best_solution[0]][2])

# 9) 可视化
best_cycle = np.append(best_solution, best_solution[0])

# 保存收敛图
plt.figure(figsize=(15, 15))
plt.plot(convergence_history, 'r-', label='历史最优')
plt.xlabel('迭代次数', fontsize=40)
plt.ylabel('路径长度', fontsize=40)
plt.legend(fontsize=40)
plt.tick_params(axis='both', labelsize=40)
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_DE\\TSP_DE_收敛历史_种群大小={POPULATION_SIZE}_迭代数={MAX_GENERATIONS}_Cr={CROSSOVER_RATE}_F={SCALING_FACTOR}.pdf', dpi=500)
plt.close()

# 保存最优路径图
plt.figure(figsize=(15, 15))
x_coords = [cities[i][0] for i in best_cycle]
y_coords = [cities[i][1] for i in best_cycle]
plt.plot(x_coords, y_coords, 'g-', linewidth=2)
plt.plot(x_coords, y_coords, 'r.', markersize=10)
for (x, y, name) in cities:
    plt.text(x * 1.001, y * 1.001, name, fontsize=25)
plt.xlabel('X坐标', fontsize=40)
plt.ylabel('Y坐标', fontsize=40)
plt.tick_params(axis='both', labelsize=40)
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_DE\\TSP_DE_最优路径_种群大小={POPULATION_SIZE}_迭代数={MAX_GENERATIONS}_Cr={CROSSOVER_RATE}_F={SCALING_FACTOR}.pdf', dpi=500)
plt.close()