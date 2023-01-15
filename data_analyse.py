#  相关包导入
from pyemd import CEEMDAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import csv

#  数据预处理
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv",'rt') as csvfile:
    reader = csv.reader(csvfile)
    close_price = [row[1] for row in reader]
# print(close_price)
print(len(close_price))
# print(close_price[1])

# 数据标准化
temp = np.array(close_price[1:len(close_price)])
Mm = MinMaxScaler()
close_price_normal = Mm.fit_transform(temp.reshape(-1, 1))

#  绘图
plt.plot(np.arange(len(close_price_normal)), close_price_normal)
plt.show()

# tips：记得设置全局变量 IImfs=[]
elec_all_day_test = []
IImfs = []
def ceemdan_decompose(data):
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    plt.figure(figsize=(12, 9))
    plt.subplots_adjust(hspace=0.1)
    plt.subplot(imfs.shape[0] + 2, 1, 1)
    plt.plot(data, 'r')
    for i in range(imfs.shape[0]):
        plt.subplot(imfs.shape[0] + 2, 1, i + 2)
        plt.plot(imfs[i], 'g')
        plt.ylabel("IMF %i" % (i + 1))
        plt.locator_params(axis='x', nbins=10)
        # 在函数前必须设置一个全局变量 IImfs=[]
        IImfs.append(imfs[i])

# 生成res的分解
def ceemdan_decompose_res(data):
    ceemdan = CEEMDAN()
    ceemdan.ceemdan(data)
    imfs, res = ceemdan.get_imfs_and_residue()
    plt.figure(figsize=(12,9))
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
    return res

# ceemdan分解
res=ceemdan_decompose_res(np.array(elec_all_day_test).ravel())

