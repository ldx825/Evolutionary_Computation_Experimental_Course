# encoding: utf-8
import random
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.rc('font', family='Microsoft YaHei')

x_init = None
score_init = 1e-9
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# 2) 读取城市坐标（文件每行：城市名 \t x \t y）
CITY_FILE = "D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\distanceMatrix.txt"
citys = []
with open(CITY_FILE, "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, x, y = line.split("\t")
        citys.append((float(x), float(y), name))

city_length = len(citys)

# 优化距离矩阵计算，使用向量化操作
cities_array = np.array([(x, y) for x, y, _ in citys])
# 计算所有城市间的欧氏距离矩阵
x_diff = cities_array[:, 0][:, np.newaxis] - cities_array[:, 0][np.newaxis, :]
y_diff = cities_array[:, 1][:, np.newaxis] - cities_array[:, 1][np.newaxis, :]
distanceCity = np.sqrt(x_diff**2 + y_diff**2)

# 3) 定义超参数
LIFE_COUNT = 100  # 蚂蚁个数
MAX_GENERATIONS = 200  # 迭代层数
init_pher = 0.001  # 初始浓度
evap_pher = 0.5  # 蒸发率
a = 1  # 信息素浓度因子
b = 3  # 启发因子

# 确保探索能力而引入最小信息素下限
MAX_pher = 1 / (evap_pher * 154.0)
MIN_pher = MAX_pher / (2.0 * city_length)

# 构建信息素矩阵
pheromone_matrix = np.full((city_length, city_length), init_pher)

def construct_ant_path():
    """构建单个蚂蚁的路径"""
    unvisited_cities = list(range(city_length))
    current_city = unvisited_cities.pop(0)  # 从城市0开始
    path = [current_city]
    total_distance = 0.0
    
    while unvisited_cities:
        # 计算下一个城市的概率
        pheromone_values = pheromone_matrix[current_city][unvisited_cities]
        heuristic_values = 1.0 / distanceCity[current_city][unvisited_cities]
        probabilities = (pheromone_values ** a) * (heuristic_values ** b)
        probabilities = probabilities / probabilities.sum()
        
        # 轮盘赌选择下一个城市
        next_city_idx = np.random.choice(len(unvisited_cities), p=probabilities)
        next_city = unvisited_cities.pop(next_city_idx)
        
        total_distance += distanceCity[current_city, next_city]
        current_city = next_city
        path.append(current_city)
    
    # 返回起始城市
    total_distance += distanceCity[current_city, 0]
    path.append(0)
    return path, total_distance

# 主循环
best_history = []
best_distance = float('inf')
best_path = []

for generation in range(MAX_GENERATIONS):
    current_best_distance = float('inf')
    current_best_path = None
    delta_pheromone = np.zeros((city_length, city_length))
    
    # 每只蚂蚁构建路径
    for _ in range(LIFE_COUNT):
        path, distance = construct_ant_path()
        
        # 更新当前迭代最优解
        if distance < current_best_distance:
            current_best_distance = distance
            current_best_path = path.copy()
        
        # 局部信息素更新（可选，原代码没有这部分）
        # 这里保持与原代码一致，只在精英蚂蚁后更新
    
    # 更新全局最优解
    if current_best_distance < best_distance:
        best_distance = current_best_distance
        best_path = current_best_path.copy()
    
    best_history.append(best_distance)
    
    # 信息素蒸发
    pheromone_matrix *= (1.0 - evap_pher)
    
    # 精英蚂蚁信息素增强
    elite_pheromone = 1.0 / best_distance
    for i in range(city_length):
        city1, city2 = best_path[i], best_path[i + 1]
        delta_pheromone[city1, city2] += elite_pheromone
        delta_pheromone[city2, city1] += elite_pheromone
    
    pheromone_matrix += delta_pheromone
    
    # 信息素边界限制
    pheromone_matrix = np.clip(pheromone_matrix, MIN_pher, MAX_pher)
    
    if generation % 10 == 0:
        print(f"经过 {generation} 次迭代，最优解距离为：{best_distance:.6f}")

# 7) 输出结果
print(f"经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{best_distance:.6f}")
print("遍历城市顺序为：")
for idx in best_path[:-1]:
    print(citys[idx][2], end=' -> ')
print(citys[best_path[0]][2])

# 8) 可视化
best_cycle = best_path + [best_path[0]]

# 保存收敛图
plt.figure(figsize=(15, 15))
plt.plot(best_history, 'r-', label='history_best')
plt.xlabel('Iteration', fontsize=40)
plt.ylabel('length', fontsize=40)
plt.legend(fontsize=40)
plt.tick_params(axis='both', labelsize=40)
# 保存收敛图单独文件
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_ACO\\TSP_ACO_convergence_history_Ants={LIFE_COUNT}_alpha={a}_beta={b}_generation={MAX_GENERATIONS}_ρ={evap_pher}.pdf', dpi=500)
plt.close()

# 保存最优路径图
plt.figure(figsize=(15, 15))
xs = [citys[i][0] for i in best_cycle]
ys = [citys[i][1] for i in best_cycle]
plt.plot(xs, ys, 'g-')
plt.plot(xs, ys, 'r.')
for (x, y, name) in citys:
    plt.text(x * 1.001, y * 1.001, name, fontsize=25)
plt.xlabel('x', fontsize=40)
plt.ylabel('y', fontsize=40)
plt.tick_params(axis='both', labelsize=40)

# 保存路径图单独文件
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_ACO\\TSP_ACO_best_path_Ants={LIFE_COUNT}_alpha={a}_beta={b}_generation={MAX_GENERATIONS}_ρ={evap_pher}.pdf', dpi=500)
plt.close()