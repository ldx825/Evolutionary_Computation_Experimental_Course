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

class Particle(object):
    """粒子类"""
    def __init__(self, position=None):
        self.position = np.array(position)
        self.velocity = np.zeros_like(position)
        self.best_position = self.position.copy()
        self.fitness = score_init
        self.best_fitness = self.fitness
        
    @property
    def gene(self):
        """从连续位置转换为离散序列"""
        return np.argsort(self.position)

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
POPULATION_SIZE = 200
MAX_GENERATIONS = 5000
C1 = 2.5
C2 = 1.0
V_MAX = 2.0
N_CITIES = len(cities)

# 4) 预计算距离矩阵以提高效率
def compute_distance_matrix():
    """预计算所有城市间的距离矩阵"""
    coords = np.array([(x, y) for x, y, _ in cities])
    dx = coords[:, 0][:, None] - coords[:, 0]
    dy = coords[:, 1][:, None] - coords[:, 1]
    return np.sqrt(dx**2 + dy**2)

distance_matrix = compute_distance_matrix()

# 5) 初始化种群
def create_particle():
    """创建单个粒子"""
    position = np.random.rand(N_CITIES)
    return Particle(position)

particles = [create_particle() for _ in range(POPULATION_SIZE)]

# 6) 定义评估函数
def evaluate_fitness(particle):
    """计算路径长度并返回适应度（路径长度的倒数）"""
    gene = particle.gene
    # 使用距离矩阵快速计算路径总长度
    idx = np.append(gene, gene[0])
    path_distances = distance_matrix[idx[:-1], idx[1:]]
    total_distance = np.sum(path_distances)
    return 1.0 / max(1e-12, total_distance)

# 7) 主循环
convergence_history = []
global_best = None
global_best_fitness = -1e9

for generation in range(MAX_GENERATIONS):
    # 更新惯性权重（线性递减）
    inertia_weight = 0.9 - 0.2 * generation / MAX_GENERATIONS
    
    # 评估所有粒子并更新个体最优
    for particle in particles:
        particle.fitness = evaluate_fitness(particle)
        
        # 更新个体历史最优
        if particle.fitness > particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_position = particle.position.copy()
        
        # 更新全局最优
        if particle.fitness > global_best_fitness:
            global_best_fitness = particle.fitness
            global_best = copy.deepcopy(particle)
    
    # 记录当前最优解的距离
    convergence_history.append(1.0 / global_best_fitness)
    
    # 更新所有粒子的速度和位置
    for particle in particles:
        # 生成随机因子
        r1, r2 = np.random.rand(N_CITIES), np.random.rand(N_CITIES)
        
        # 更新速度
        cognitive_component = C1 * r1 * (particle.best_position - particle.position)
        social_component = C2 * r2 * (global_best.position - particle.position)
        particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component
        
        # 限制速度范围
        particle.velocity = np.clip(particle.velocity, -V_MAX, V_MAX)
        
        # 更新位置
        particle.position += particle.velocity
    
    # 可选：定期输出进度
    if generation % 500 == 0:
        current_best_distance = 1.0 / global_best_fitness
        print(f"迭代 {generation:4d} 代，当前最优距离: {current_best_distance:.6f}")

# 8) 输出结果
final_distance = 1.0 / global_best_fitness
print(f"\n经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_distance:.6f}")
print("遍历城市顺序为：")
best_gene = global_best.gene
for idx in best_gene:
    print(cities[idx][2], end=' -> ')
print(cities[best_gene[0]][2])

# 9) 可视化
best_cycle = np.append(best_gene, best_gene[0])

# 保存收敛图
plt.figure(figsize=(15, 15))
plt.plot(convergence_history, 'r-', label='历史最优')
plt.xlabel('迭代次数', fontsize=40)
plt.ylabel('路径长度', fontsize=40)
plt.legend(fontsize=40)
plt.tick_params(axis='both', labelsize=40)
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_PSO\\TSP_PSO_收敛历史_粒子数={POPULATION_SIZE}_迭代数={MAX_GENERATIONS}_v最大={V_MAX}_c1={C1}_c2={C2}.pdf', dpi=500)
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
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_PSO\\TSP_PSO_最优路径_粒子数={POPULATION_SIZE}_迭代数={MAX_GENERATIONS}_v最大={V_MAX}_c1={C1}_c2={C2}.pdf', dpi=500)
plt.close()