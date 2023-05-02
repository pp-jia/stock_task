# 相关包导入
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# 数据及处理
data = pd.read_csv(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv")  # 训练
data = data.drop(['Unnamed: 0'], axis=1)

# 参数
pop_size = 30  # 种群数量
max_value = 10  # 基因中允许出现的最大值
chromosomal_length = 14  # 染色体长度  它的长度是特征个数
pc = 0.6  # 交配概率
pm = 0.01  # 变异率
results = []  # 存储每一代的最优解，N个二元组
fit_value = []  # 个体适应度
fit_mean = []  # 平均适应度

#初始化种群
def geneEncoding(pop_size, chromosomal_length):
    pop = [[]]
    for i in range(pop_size):
        temp = []
        for j in range(chromosomal_length):
            temp.append(random.randint(0, 1))
        pop.append(temp)
    return pop[1:]
pop = geneEncoding(pop_size, chromosomal_length)

# 提取data和label
data_X = data.drop(['close'], axis=1)
data_Y = data.close
data_Y = data_Y.astype('int')
feature_name = data_X.columns


# 计算适应度
import sklearn
from sklearn import tree
from sklearn.model_selection import cross_val_score


# 计算适应度，使用决策树的准确率作为适应度   （决策树用于分类问题，这里我们采用相关系数来做适应度函数）
def calFitness(pop, chrom_length, max_value, data_X, data_Y, feature_names):
    obj_value = []
    for i in range(len(pop)):
        data_X_test = data_X
        for j in range(len(pop[i])):
            if pop[i][j] == 0:
                data_X_test = data_X_test.drop([feature_names[j]], axis=1)
        # 决策树
        # clf = tree.DecisionTreeClassifier()
        # score = cross_val_score(clf, data_X_test, data_Y, cv=5, scoring='f1').mean()
        # print(np.shape(data_X_test), np.shape(data_Y))
        # 这样的处理是否合理？
        score = sklearn.metrics.r2_score(np.mean(data_X_test, axis=1), data_Y, sample_weight=None, multioutput='uniform_average')
        obj_value.append(score)
    return obj_value

# 种群选择
def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total

def cumsum(fit_value):
    for i in range(len(fit_value)-2, -1, -1):
        t = 0
        j = 0
        while(j <= i):
            t += fit_value[j]
            j += 1
        fit_value[i] = t
        fit_value[len(fit_value)-1] = 1

def selection(pop, fit_value):
    newfit_value = []
    # 适应度总和
    total_fit = sum(fit_value)
    for i in range(len(fit_value)):
        newfit_value.append(fit_value[i] / total_fit)
    # 计算累计概率
    cumsum(newfit_value)
    ms = []
    for i in range(len(pop)):
        ms.append(random.random())
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    # 轮转盘算法
    while newin < len(pop):
        if(ms[newin] < newfit_value[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop

# 交配
def crossover(pop, pc):
    for i in range(len(pop) - 1):
        if(random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i+1][cpoint:len(pop[i])])
            temp2.extend(pop[i+1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            pop[i] = temp1
            pop[i+1] = temp2
# 变异
def mutation(pop, pm):
    px = len(pop)
    py = len(pop[0])

    for i in range(px):
        if(random.random() < pm):
            mpoint = random.randint(0, py-1)
            if(pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            else:
                pop[i][mpoint] = 1

# 最优解
def best(pop, fit_value):
    px = len(pop)
    best_individual =[]
    best_fit = fit_value[0]
    for i in range(1, px):
        if(fit_value[i] > best_fit):
            best_fit = fit_value[i]
            best_individual = pop[i]
    return [best_individual, best_fit]

def mean(obj_value):
    return sum(obj_value) / len(obj_value)
for i in range(50):
    obj_value = calFitness(pop, chromosomal_length, max_value, data_X, data_Y, feature_name) # 个体评价

    best_individual, best_fit = best(pop, obj_value) # 储存最优解和最优基因
    results.append([best_individual, best_fit, mean(obj_value)])

    selection(pop, obj_value)  # 新种群复制
    crossover(pop, pc)  # 交配
    mutation(pop, pm)  # 变异
results = results[1:]
results.sort()
print(results)

X = []
Y = []
for i in range(49):
    X.append(i)
    t = results[i][0]
    Y.append(t)
plt.plot(X, Y)
plt.show()

