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
        # self.v = np.zeros_like(x)
        # 消融实验：速度的初始化!
        if x is not None:
            self.v = np.random.uniform(-0.5, 0.5, size=len(x))
        else:
            self.v = np.array([])  # 或 np.zeros(0)
        self.pbest = self.x.copy()
        self.score = score_init
        self.pbest_score = self.score

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# 2) 读取城市坐标（文件每行：城市名 \t x \t y）
# CITY_FILE = "cn34.txt"
CITY_FILE = "D:\\大二上-吉大相关\\演化计算代码作业\\Evolutionary_Computation_Experimental_Course\\distanceMatrix.txt"
citys = []
with open(CITY_FILE, "r", encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        name, x, y = line.split("\t")
        citys.append((float(x), float(y), name)) # 😁

# 3) 定义超参数
LIFE_COUNT = 200
MAX_GENERATIONS = 5000
# 消融实验：
C1 = 2.5
# C1 = 0.0
C2 = 1.0
# 消融实验：
# C2 = 0.0
V_MAX = 2.0
# 消融实验：速度没有限制！
# V_MAX = float('inf')
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

# 6) 主循环
best_history = []
gbest = Life(x_init)
gbest.score = -1e9

for gen in range(MAX_GENERATIONS):
    
    new_lives = lives[:]
    # 计算适应度
    for life in new_lives:
        life.score = evaluate(life)
        if life.score > gbest.score:
            gbest = copy.deepcopy(life)
        if life.score > life.pbest_score:
            life.pbest_score = life.score
            life.pbest = life.x.copy()

    # 保存最佳个体分数
    best_history.append(1.0 / gbest.score)

    # 更新速度
    for life in new_lives:
        R1 = np.random.rand(gene_length)
        R2 = np.random.rand(gene_length)
        W = 0.9  - 0.2 * gen / MAX_GENERATIONS
        # 消融实验：完全没有惯性信息！！
        # W = 0.0
        # W = 0.8
        life.v = W * life.v + C1 * R1 * (life.pbest - life.x)  + C2 * R2 *  (gbest.x - life.x)
        life.v = np.clip(life.v, -V_MAX, V_MAX)
        life.x = life.v +life.x

    # 我们的x现在是连续的，我们要将之合理的转化为离散的序列
    # 转化为合理的gene(x)
    for life in new_lives:
        life.gene = np.argsort(life.x)

    # 更新种群
    lives = new_lives[:]

# 7) 输出结果
final_best_distance = 1.0 / gbest.score
print(f"经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_best_distance:.6f}")
print("遍历城市顺序为：")
for idx in gbest.gene:
    print(citys[idx][2], end=' -> ')
print(citys[gbest.gene[0]][2])

# 8) 可视化
best_cycle = list(gbest.gene[:]) + list([gbest.gene[0]])

# 保存收敛图
plt.figure(figsize=(15, 15))
plt.plot(best_history, 'r-', label='history_best')
plt.xlabel('Iteration', fontsize=40)
plt.ylabel('length', fontsize=40)
plt.legend(fontsize=40)
plt.tick_params(axis='both', labelsize=40)
# 保存收敛图单独文件
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\Evolutionary_Computation_Experimental_Course\\TSP_PSO\\Ablation_TSP_PSO_convergence_history_Particles={LIFE_COUNT}_generation={MAX_GENERATIONS}_v_max={V_MAX}_c_1={C1}_c_2={C2}.pdf', dpi=500)
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
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\Evolutionary_Computation_Experimental_Course\\TSP_PSO\\Ablation_TSP_PSO_best_path_Particles={LIFE_COUNT}_generation={MAX_GENERATIONS}_v_max={V_MAX}_c_1={C1}_c_2={C2}.pdf', dpi=500)
plt.close()