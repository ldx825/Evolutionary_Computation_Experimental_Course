# encoding: utf-8
import random
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft YaHei')

SCORE_NONE = -1

class Life(object):
    """个体类"""
    def __init__(self, aGene=None):
        self.gene = aGene
        self.score = SCORE_NONE  # 适配值/得分

random.seed(42) 

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

# 3) 定义超参数
LIFE_COUNT = 150
CROSS_RATE = 0.9
MUTATION_RATE = 0.05
MAX_GENERATIONS = 2000

# 4) 初始化种群
gene_length = len(citys)
base = list(range(gene_length))
lives = [Life(random.sample(base, len(base))) for _ in range(LIFE_COUNT)]

# 5) 定义评估函数（回路距离的倒数）
def evaluate(life):
    dist = 0.0
    for i in range(gene_length):
        i1, i2 = life.gene[i], life.gene[(i + 1) % gene_length]
        x1, y1, _ = citys[i1]
        x2, y2, _ = citys[i2]
        dist += math.hypot(x1 - x2, y1 - y2)
    return 1.0 / max(1e-12, dist)

def pmx_crossover(parent1, parent2, a, b):
    """部分映射交叉"""
    child1, child2 = parent1.gene[:], parent2.gene[:]
    
    for j in range(a, b + 1):
        fa, fb = child1[j], child2[j]
        
        # 交换两个位置的基因
        idx1, idx2 = child1.index(fb), child2.index(fa)
        child1[idx1], child2[idx2] = fa, fb
        child1[j], child2[j] = fb, fa
    
    return Life(child1), Life(child2)

def ox_crossover(parent1, parent2, a, b):
    """顺序交叉"""
    list1, list2 = parent1.gene[a:b], parent2.gene[a:b]
    
    left1 = [x for x in parent2.gene if x not in list1]
    left2 = [x for x in parent1.gene if x not in list2]
    
    return Life(left1[:a] + list1 + left1[a:]), Life(left2[:a] + list2 + left2[a:])

def swap_mutation(gene):
    """交换变异"""
    a, b = random.sample(range(gene_length), 2)
    gene[a], gene[b] = gene[b], gene[a]
    return gene

def reverse_mutation(gene):
    """反序变异"""
    a, b = sorted(random.sample(range(gene_length), 2))
    gene[a:b] = reversed(gene[a:b])
    return gene

def insert_mutation(gene):
    """插入变异"""
    a, b = random.sample(range(gene_length), 2)
    c = gene.pop(b)
    gene.insert(a, c)
    return gene

# 6) GA 主循环
best_history = []
best = None

for gen in range(MAX_GENERATIONS):
    # 计算适应度并找到最优个体
    for life in lives:
        life.score = evaluate(life)
    
    # 获取当前最优个体
    current_best = max(lives, key=lambda x: x.score)
    best_history.append(1.0 / current_best.score)
    
    # 更新全局最优
    if best is None or current_best.score > best.score:
        best = Life(current_best.gene[:])
    
    # 选择 - 轮盘赌
    sum_score = sum(life.score for life in lives)
    selected_lives = []
    
    for _ in range(int(LIFE_COUNT * CROSS_RATE)):
        r = random.uniform(0, sum_score)
        cumulative = 0
        for life in lives:
            cumulative += life.score
            if cumulative > r:
                selected_lives.append(Life(life.gene[:]))
                break
    
    # 交配
    mated_lives = []
    random.shuffle(selected_lives)
    
    while len(selected_lives) >= 2:
        parent1 = selected_lives.pop()
        parent2 = selected_lives.pop()
        a, b = sorted(random.sample(range(gene_length), 2))
        
        if random.random() < 0.33:  # 部分映射交叉
            child1, child2 = pmx_crossover(parent1, parent2, a, b)
        else:  # 顺序交叉
            child1, child2 = ox_crossover(parent1, parent2, a, b)
        
        mated_lives.extend([child1, child2])
    
    # 变异
    for life in mated_lives:
        if random.random() < MUTATION_RATE:
            mutation_type = random.choice(['swap', 'reverse', 'insert'])
            
            if mutation_type == 'swap':
                life.gene = swap_mutation(life.gene[:])
            elif mutation_type == 'reverse':
                life.gene = reverse_mutation(life.gene[:])
            else:
                life.gene = insert_mutation(life.gene[:])
    
    # 锦标赛选择生成下一代
    next_lives = [Life(best.gene[:])]  # 保留最优个体
    
    for _ in range(LIFE_COUNT - 1):
        a, b = random.sample(lives + mated_lives, 2)
        winner = a if a.score > b.score else b
        next_lives.append(Life(winner.gene[:]))
    
    # 更新种群
    lives = next_lives

# 7) 输出结果
final_best_distance = 1.0 / best.score
print(f"经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_best_distance:.6f}")
print("遍历城市顺序为：")
for idx in best.gene:
    print(citys[idx][2], end=' -> ')
print(citys[best.gene[0]][2])

# 8) 可视化
best_cycle = best.gene[:] + [best.gene[0]]

# 保存收敛图
plt.figure(figsize=(15, 15))
plt.plot(best_history, 'r-', label='history_best')
plt.xlabel('Iteration', fontsize=40)
plt.ylabel('length', fontsize=40)
plt.legend(fontsize=40)
plt.tick_params(axis='both', labelsize=40)
# 保存收敛图单独文件
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_GA\\TSP_GA_convergence_history_NP={LIFE_COUNT}_Generation={MAX_GENERATIONS}_CR={CROSS_RATE}_MR={MUTATION_RATE}.pdf', dpi=500)
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
plt.savefig(f'D:\\大二上-吉大相关\\演化计算代码作业\\TSP_all\\TSP_GA\\TSP_GA_best_path_NP={LIFE_COUNT}_Generation={MAX_GENERATIONS}_CR={CROSS_RATE}_MR={MUTATION_RATE}.pdf', dpi=500)
plt.close()