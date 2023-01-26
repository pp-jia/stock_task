import pickle
import numpy as np

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

# 加载变量
train_predict = load_variable(r"C:\Users\31269\Desktop\毕设\variable\train_predict.txt")
test_predict = load_variable(r"C:\Users\31269\Desktop\毕设\variable\test_predict.txt")
print(np.shape(train_predict))
print(np.shape(test_predict))

train_true = low_f[30:158310+30]
test_true = low_f[158310 + 30 * 2 - 1:len(low_f)-1]
print(np.shape(train_true))
print(np.shape(test_true))


def hit_rate(true, predict):
    DS_sum = 0
    for i in range(len(true)-1):
        if ((predict[i+1]-predict[i]) * (true[i+1]-true[i])) >= 0:
            temp = 1
            DS_sum += temp
        else:
            temp = 0
            DS_sum += temp
    DS = DS_sum / len(true)
    return DS

train_hit_rate = hit_rate(train_true, train_predict)
test_hit_rate = hit_rate(test_true, test_predict)

print(train_hit_rate)
print(test_hit_rate)

