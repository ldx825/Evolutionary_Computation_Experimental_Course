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
# CITY_FILE = "att48.txt"
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

distanceCity = np.zeros((city_length, city_length))
for i in range(city_length):
    for j in range(i, city_length):
        x1, y1, _ = citys[i]
        x2, y2, _ = citys[j]
        distanceCity[i][j] = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        distanceCity[j][i] = distanceCity[i][j]

# 3) 定义超参数
LIFE_COUNT = 100 # 蚂蚁个数
MAX_GENERATIONS = 200 # 迭代层数
init_pher = 0.001 # 初始浓度
evap_pher = 0.5 # 蒸发率
a = 1 # 信息素浓度因子
b = 3 # 启发因子
# 确保探索能力而引入最小信息素下限
MAX_pher = 1 / (evap_pher * 154.0)
MIN_pher = MAX_pher / (2.0 * city_length)

# 构建矩阵
Array = np.full((city_length, city_length), init_pher) # 用于保存现在的信息素矩阵
# 主循环
best_history = []
best = 1000
best_route = []

for gen in range(MAX_GENERATIONS):
    
    Add_Array = np.zeros((city_length, city_length)) # 用于记录当前一代蚂蚁走过的路累加的信息素

    for life in range(LIFE_COUNT):
        life_length = 0
        route = []
        leave_city = [i for i in range(city_length)]
        route.append(0)
        now_city = leave_city.pop(0)

        # 外层循环完成city_length - 1次选择，每次从尚未到达的城市中选择一个
        for t in range(city_length - 1):
            # 根据伪随机比例规则构建概率向量
            sum_RP = 0
            row_RP = []

            # 内层循环完成在没有到过的城市中选择一个作为下一个城市的概率向量
            for g in leave_city:
                now_RP = (Array[now_city][g] ** a) * ((1 / distanceCity[now_city][g]) ** b)
                sum_RP += now_RP
                now_RP = sum_RP
                row_RP.append(now_RP)
            true_RP = [r/ sum_RP for r in row_RP ]
            # 使用轮赌法完成城市选取
            random_num = random.random()
            for i in range(len(true_RP)):
                if random_num < true_RP[i]:
                    next_city = leave_city.pop(i)
                    life_length += distanceCity[now_city][next_city]
                    now_city = next_city
                    break
            else:
                next_city = leave_city.pop(-1) # 弹出最后一个
                life_length += distanceCity[now_city][next_city]
                now_city = next_city    
            
            route.append(now_city)
        
        route.append(0)
        life_length += distanceCity[now_city][0]

        # 比较记录最短路径和最短路径序列
        if life_length < best :
            best = life_length
            best_route = route.copy()
    best_history.append(best)

    Array = Array*(1 - evap_pher)

    # 精英蚂蚁机制
    add_pher = 1 / best
    for i in range(city_length):
        i1 = best_route[i]
        i2 = best_route[i+1]
        Add_Array[i1][i2] += add_pher
        Add_Array[i2][i1] += add_pher

    Array += Add_Array

    # 确保探索能力而引入最小信息素下限
    for i in range(city_length):
        for j in range(city_length):
            Array[i][j] = max(Array[i][j], MIN_pher)
            Array[i][j] = min(Array[i][j], MAX_pher)
    if gen % 10 == 0:
        print(f"经过 {gen} 次迭代，最优解距离为：{best:.6f}")


# 7) 输出结果
final_best_distance = best
print(f"经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_best_distance:.6f}")
print("遍历城市顺序为：")
for idx in best_route[:-1]:
    print(citys[idx][2], end=' -> ')
print(citys[best_route[0]][2])

# 8) 可视化
best_cycle = list(best_route[:]) + list([best_route[0]])

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