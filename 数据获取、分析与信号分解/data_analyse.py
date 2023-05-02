#  相关包导入
from PyEMD import CEEMDAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import csv
import warnings
import pickle   # 用于保存变量的相关包
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

#  数据预处理  输入your数据路径
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    close_price = [row[1] for row in reader]
# print(close_price)
print(len(close_price))
# print(close_price[1])

# 数据标准化
temp = np.array(close_price[1:len(close_price)])
print(np.shape(temp))
# print(np.shape(temp.reshape(-1, 1)))
Stand = StandardScaler()
close_price_normal = Stand.fit_transform(temp.reshape(-1, 1)).T
print(np.shape(close_price_normal))
# print(close_price_normal[0])   ##数据输入只能是一维的.
print(type(close_price_normal))

#  绘图
plt.plot(np.arange(len(close_price_normal[0])), close_price_normal[0])
plt.show()

# tips：记得设置全局变量 IImfs=[]
IImfs = []
# 生成res的分解
def ceemdan_decompose_res(data):
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    plt.figure(figsize=(12,10))
    plt.subplots_adjust(hspace=0.1)
    plt.subplot(imfs.shape[0]+3, 1, 1)
    plt.plot(data,'r')
    for i in range(imfs.shape[0]):
        plt.subplot(imfs.shape[0]+3,1,i+2)
        plt.plot(imfs[i], 'g')
        plt.ylabel("IMF %i" %(i+1))
        plt.locator_params(axis='x', nbins=10)
        # 在函数前必须设置一个全局变量 IImfs=[]
        IImfs.append(imfs[i])
    plt.subplot(imfs.shape[0]+3, 1, imfs.shape[0]+3)
    plt.plot(res,'g')
    plt.show()
    return res
# ceemdan分解

res = ceemdan_decompose_res(close_price_normal[0])

# print(np.array(IImfs))
# print(res)

# 变量保存函数
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

# 保存变量
filename = save_variable(IImfs, r"C:\Users\31269\Desktop\毕设\variable\IImfs.txt")
