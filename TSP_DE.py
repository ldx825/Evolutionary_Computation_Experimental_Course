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

class Life(object):
    """个体类"""
    def __init__(self, x=None):
        self.x = np.array(x)
        self.gene = np.argsort(x)
        self.score = score_init

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 2) 读取城市坐标（文件每行：城市名 \t x \t y）
CITY_FILE = "D:\\大二上-吉大相关\\演化计算代码作业\\Evolutionary_Computation_Experimental_Course\\distanceMatrix.txt"
citys = []
with open(CITY_FILE, "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, x, y = line.split("\t")
        citys.append((float(x), float(y), name))

# 3) 定义超参数
LIFE_COUNT = 120
MAX_GENERATIONS = 2000
# MAX_GENERATIONS = 1
Cr = 0.9
# 消融实验
# Cr = 0.0
F = 0.6
# 消融实验
# F = 0.0

gene_length = len(citys)

# 4) 初始化种群
lives = []
base = list(range(gene_length))
for i in range(LIFE_COUNT):
    x = [random.random() for _ in range(gene_length)]
    lives.append(Life(x))

# 5) 定义评估函数（回路距离的倒数）
def evaluate(life):
    dist = 0.0
    for i in range(gene_length):
        i1 = life.gene[i]
        i2 = life.gene[(i + 1)%gene_length]
        x1, y1, _ = citys[i1]
        x2, y2, _ = citys[i2]
        dist += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return 1.0 / max(1e-12, dist)
    # return 1.0

# 6) 主循环
best_history = []
best_value = 0
best_gene = []

for gen in range(MAX_GENERATIONS):
    
    new_lives = lives[:]

    # 计算适应度
    for life in new_lives:
        life.score = evaluate(life)
        if life.score > best_value:
            best_value = life.score
            best_gene = life.gene

    # 保存最佳个体分数
    best_history.append(1.0 / best_value)

    # 更新速度
    for life in new_lives:
        R0 = np.random.randint(0, LIFE_COUNT)
        R1 = 0
        R2 = 0
        u = [random.random() for _ in range(gene_length)]
        while True:
            R1 = np.random.randint(0, LIFE_COUNT)
            if R1 == R0:
                continue
            else:
                break
        while True:
            R2 = np.random.randint(0, LIFE_COUNT)
            if R2 == R0 or R2 == R1:
                continue
            else:
                break
        
        jrand = np.random.randint(0,gene_length)


        for j in range(gene_length) :
            if random.random() <= Cr or j == jrand:
                u[j] = life.x[j] + F * (lives[R1].x[j] - lives[R2].x[j])
            else:
                u[j] = life.x[j]

        U = Life(u)

        # if evaluate(U) > evaluate(life):
        #     life.x = u
        #     life.gene = np.argsort(life.x)
        # 消融实验：完全随机选择
        if random.random() < 0.5:
            life.x = u
            life.gene = np.argsort(life.x)

    # 更新种群
    lives = new_lives[:]

    if gen % 500 == 0:
        print(f"经过了{gen}代之后，最优解为{ 1 / best_value}")

# 7) 输出结果
final_best_distance = 1.0 / best_value
print(f"经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_best_distance:.6f}")
print("遍历城市顺序为：")
for idx in best_gene:
    print(citys[idx][2], end=' -> ')
print(citys[best_gene[0]][2])

# 8) 可视化
best_cycle = list(best_gene[:]) + list([best_gene[0]])

# 保存收敛图
plt.figure(figsize=(15, 15))
plt.plot(best_history, 'r-', label='history_best')
plt.xlabel('Iteration', fontsize=40)
plt.ylabel('length', fontsize=40)
plt.legend(fontsize=40)
plt.tick_params(axis='both',labelsize=40)
# 保存收敛图单独文件
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\Evolutionary_Computation_Experimental_Course\\TSP_DE\\Ablation_TSP_DE_convergence_history_NP={LIFE_COUNT}_generation={MAX_GENERATIONS}_Cr={Cr}_F={F}.pdf', dpi=500)
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
plt.tick_params(axis='both',labelsize=40)

# 保存路径图单独文件
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\Evolutionary_Computation_Experimental_Course\\TSP_DE\\Ablation_TSP_DE_best_path_NP={LIFE_COUNT}_generation={MAX_GENERATIONS}_Cr={Cr}_F={F}.pdf', dpi=500)
plt.close()