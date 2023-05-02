import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 变量加载函数
def load_variable(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r
# 加载变量
IImfs = load_variable(r"C:\Users\31269\Desktop\毕设\variable\IImfs.txt")
# print(np.shape(IImfs))
low_f = IImfs[11] + IImfs[12] + IImfs[13] + IImfs[14] + IImfs[15]
print(np.shape(low_f))

# 数据标准化
Stand = StandardScaler()
def normalize(data):
    data = np.array(data[1:len(data)])
    data = np.array(data)
    data = Stand.fit_transform(data.reshape(-1, 1)).T
    return data
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    close_price = [row[1] for row in reader]
close_price_normal = normalize(close_price)

# 加载变量
test_predict = load_variable(r"C:\Users\31269\Desktop\毕设\variable\transformer_test_pred.txt")
print(np.shape(test_predict))

print(np.shape(close_price_normal))
print(len(low_f))
test_true = close_price_normal[0, 30 + int(len(low_f) * 0.7) + int(len(low_f) * 0.2)-1:len(low_f)-1]
print(np.shape(test_true))

print(test_true)

def hit_rate(true, predict):
    DS_sum = 0
    print(len(true))
    print(len(predict))
    for i in range(len(predict)-1):
        # print((predict[i+1]-predict[i]) * (true[i+1]-true[i]))
        print("------")
        print(predict[i+1])
        print("------")
        # print(true[i+1])
        if ((predict[i+1]-predict[i]) * (true[i+1]-true[i])) >= 0:
            temp = 1
            DS_sum += temp
        else:
            temp = 0
            DS_sum += temp
    DS = DS_sum / 17467
    return DS

test_hit_rate = hit_rate(test_true, test_predict)

print(len(test_true))
print(len(test_predict))
plt.plot(test_true[0:len(test_predict)], color='red', label='real')
plt.plot(test_predict[0:len(test_predict)], color='blue', label='predict')
plt.legend()
plt.show()

print(test_hit_rate)