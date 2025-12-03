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
lives = []
base = list(range(gene_length))
for i in range(LIFE_COUNT):
    g = base[:]
    random.shuffle(g)
    lives.append(Life(g))

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

# 6) GA 主循环
best_history = []
best = None

for gen in range(MAX_GENERATIONS):
    new_lives = lives[:]

    next_lives = []
    # 计算适应度
    best_score = 0
    sum_score = 0
    for life in new_lives:
        life.score = evaluate(life)
        sum_score += life.score
        if life.score > best_score:
            best_score = life.score

    best_history.append(1.0 / best_score)
    # 保留最优个体
    for life in new_lives:
        if life.score == best_score:
            best = life

    # 选择
    selected_lives = []
    for j in range(int(LIFE_COUNT * CROSS_RATE )):
        i = 0
        random_num = random.uniform(0, sum_score)
        for life in new_lives:
            i += life.score
            if random_num < i :
                selected_lives.append(Life(life.gene[:]))
                break
    
    # 交配
    mated_lives =[]
    while len(selected_lives) >1 :
        o = random.randint(0,2)
        
        parent1 = selected_lives.pop()
        parent2 = selected_lives.pop()

        a = random.randint(0,33)
        b = random.randint(0,33)

        if a > b:   # 确保a小于b
            t = a
            a = b
            b = t

        # 进行部分映射交叉（1/3）的概率
        if o == 1:

            for j in range(a, b+1):
                fa = parent1.gene[j] 
                fb = parent2.gene[j]
                for i in range(0,gene_length):
                    if parent1.gene[i] == fb:
                        parent1.gene[i] = fa
                parent1.gene[j] = fb
                for i in range(0,gene_length):
                    if parent2.gene[i] == fa:
                        parent2.gene[i] = fb
                parent2.gene[j] = fa
            
        # 进行顺序交叉
        else:
            list1 = parent1.gene[a:b]
            list2 = parent2.gene[a:b]
            left_list1 = [x for x in parent2.gene if x not in list1]
            left_list2 = [x for x in parent1.gene if x not in list2]

            parent1.gene = left_list1[:a] + list1 + left_list1[a:]
            parent2.gene = left_list2[:a] + list2 + left_list2[a:]

        mated_lives.append(parent1)
        mated_lives.append(parent2)


    # 变异
    for life in mated_lives:
        k = random.random()
        if k < MUTATION_RATE:
            t = random.randint(1,3)
            # 交换变异
            if t == 1:
                a = random.randint(0,33)
                b = random.randint(0,33)
                t = life.gene[a]
                life.gene[a] = life.gene[b]
                life.gene[b] = t
            # 反序变异
            if t == 2:
                a = random.randint(0,33)
                b = random.randint(0,33)
                list1 = life.gene[a:b]
                list1.reverse()
                life.gene[a:b] = list1
            # 插入变异
            if t == 3:
                a = random.randint(0,33)
                b = random.randint(0,33)
                c = life.gene.pop(b)
                life.gene.insert(a,c)
    
        
    # 重新评估适应度
    for life in mated_lives:
        life.score = evaluate(life)
        sum_score += life.score

    # # 使用轮盘赌更新个体
    # next_lives.append(Life(best.gene[:])) # 这是之前保留的最佳个体
    # new_lives += mated_lives    # 合并父子代，取得
    # for j in range(LIFE_COUNT - 1 ):
    #     i = 0
    #     random_num = random.uniform(0, sum_score)
    #     for life in new_lives:
    #         i += life.score
    #         if random_num < i :
    #             next_lives.append(Life(life.gene[:]))
    #             break

    # 使用锦标赛更新个体
    next_lives.append(Life(best.gene[:])) # 这是之前保留的最佳个体
    new_lives += mated_lives
    for j in range(LIFE_COUNT - 1 ):
        a, b = random.sample(new_lives,k = 2)
        if a.score > b.score :
            next_lives.append(Life(a.gene[:]))
        else:
            next_lives.append(Life(b.gene[:]))
    # 更新种群
    lives = next_lives[:]

# 7) 输出结果
final_best_distance = 1.0 / best.score
print(f"经过 {MAX_GENERATIONS} 次迭代，最优解距离为：{final_best_distance:.6f}")
print("遍历城市顺序为：")
for idx in best.gene:
    print(citys[idx][2], end=' -> ')
print(citys[best.gene[0]][2])

# 8) 可视化
best_cycle = list(best.gene[:]) + list([best.gene[0]])

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