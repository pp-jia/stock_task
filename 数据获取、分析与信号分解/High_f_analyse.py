import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler

# 防止绘制出来的图标题/图例 显示中文异常
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

#  导入开盘价、最高价、最低价、收盘价
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    close_price = [row[1] for row in reader]
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    open_price = [row[2] for row in reader]
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    high_price = [row[3] for row in reader]
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    low_price = [row[4] for row in reader]

# 数据标准化
Stand = StandardScaler()
def normalize(data):
    data = np.array(data[1:len(data)])
    data = np.array(data)
    data = Stand.fit_transform(data.reshape(-1, 1)).T
    return data
close_price_normal = normalize(close_price)
open_price_normal = normalize(open_price)
high_price_normal = normalize(high_price)
low_price_normal = normalize(low_price)

# 变量加载函数
def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r
# 加载变量
IImfs = load_variable(r"C:\Users\31269\Desktop\毕设\variable\IImfs.txt")
# print(np.shape(IImfs))
high_f = IImfs[0] + IImfs[1] + IImfs[2] + IImfs[3] + IImfs[4] + IImfs[5] + IImfs[6]  # IImfs[7] + IImfs[8] + IImfs[9] + IImfs[10]
print(np.shape(high_f))

print(np.shape(open_price))
# print(high_f.reshape(-1, 1).T)
DT_data = np.concatenate((high_f.reshape(-1, 1).T, open_price_normal, high_price_normal, low_price_normal), axis=0)
print(np.shape(DT_data))

#  绘图
fig, ax = plt.subplots()
ax.plot(np.arange(len(close_price_normal[0])), close_price_normal.T, label='original')
ax.plot(np.arange(len(high_f)), high_f, label='high_f')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title('IImfs分解后高频与原始信号对比')  # 设置图名
ax.legend()  # 自动检测要在图例中显示的元素，并且显示
plt.show()

# Dual Thrust策略
N = 5
k1 = 0.2
k2 = 0.2

signal = []

for i in range(len(DT_data[0])):
    if i <= 5 | i == len(DT_data[0]):
        signal.append("不交易")
    else:
        curren_open = DT_data[1][i]
        HH = np.max(DT_data[2][i-6:i])    # 前N天最高价
        HC = np.max(DT_data[0][i-6:i])   # 前N天最高收盘价
        LC = np.max(DT_data[0][i-6:i])     # 前N天最低收盘价
        LL = np.max(DT_data[3][i-6:i])   # 前N天最低价
        Range = max(HH - LC, HC, LL)
        DT_buy_line = curren_open + Range * k1   # 上轨线
        DT_sell_line = curren_open - Range * k2  # 下轨线

        if DT_data[0][i+1] > DT_buy_line:
            signal.append("买")
        if DT_data[0][i+1] < DT_sell_line:
            signal.append("卖")
print(signal)

