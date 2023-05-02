# 续data_analyse
import math
import pickle   # 用于保存变量的相关包
import numpy as np
import csv
import matplotlib.pyplot as plt
import warnings

from keras import optimizers
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers import LSTM, Dropout
from sklearn.metrics import mean_squared_error
# 防止绘制出来的图标题/图例 显示中文异常
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

#  导入开盘价、最高价、最低价、换手率、成交量等数据
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
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    volume = [row[5] for row in reader]
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    chg = [row[7] for row in reader]
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
volume_normal = normalize(volume)
chg_normal = normalize(chg)

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
# low_f = IImfs[5] + IImfs[6] + IImfs[7] + IImfs[8] + IImfs[9] + IImfs[10]
# print(open_price)

print(np.shape(open_price))
print(low_f.reshape(-1, 1).T)
all_data = np.concatenate((low_f.reshape(-1, 1).T, open_price_normal, high_price_normal, low_price_normal, volume_normal, chg_normal), axis=0)
print(np.shape(all_data))

#  绘图
fig, ax = plt.subplots()
ax.plot(np.arange(len(close_price_normal[0])), close_price_normal.T, label='original')
ax.plot(np.arange(len(low_f)), low_f, label='middle_f')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.set_title('IImfs分解后中频与原始信号对比')  # 设置图名
ax.legend()  # 自动检测要在图例中显示的元素，并且显示
plt.show()


# LSTM部分开始
# 根据原始数据集构建矩阵
def create_dataset(data, time_steps, true_y):
    dataX, dataY = [], []
    # print("此时的inputshape：{}".format(len(data[1])))
    for i in range(len(data[1]) - time_steps):
        a = data[:, i:(i + time_steps)]
        dataX.append(a)
        dataY.append(true_y[0, i + time_steps])
    return np.array(dataX), np.array(dataY)

# 切割为训练集和测试集
# print(len(low_f))
train_size = int(len(low_f) * 0.9)
test_size = len(low_f) - train_size
train, test = all_data[:, 0:train_size], all_data[:, train_size:len(low_f)]
# print(np.shape(close_price_normal))
train_y = close_price_normal[:, 0:train_size]
test_y = close_price_normal[:, train_size:len(low_f)]
time_steps = 30
trainX, trainY = create_dataset(train, time_steps, train_y)
testX, testY = create_dataset(test, time_steps, test_y)


# reshape输入模型数据的格式为：[samples, time steps, features]
print(np.shape(trainX))
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[2], 6))
testX = np.reshape(testX, (testX.shape[0], testX.shape[2], 6))

# LSTM模型构建
model = Sequential()
# model.add()
model.add(GRU(128, input_shape=(time_steps, 6)))
model.add(Dropout(0.2))
model.add(Dense(1))
adam = optimizers.Adam(lr=0.001)  # decay是学习率衰减
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()
history = model.fit(trainX, trainY, epochs=20, batch_size=128, verbose=1)
score = model.evaluate(testX, testY, batch_size=64, verbose=1)

# loss曲线绘制
def visualize_loss(history, title):
    loss = history.history["loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
visualize_loss(history, "Training Loss")

# 预测训练集与测试集
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# # 对预测结果进行反归一化处理
# trainPredict = Stand.inverse_transform(trainPredict)
# trainY = Stand.inverse_transform([trainY])
# testPredict = Stand.inverse_transform(testPredict)
# testY = Stand.inverse_transform([testY])

# 计算训练集与测试集的RMSE
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:]))
print('Test Score: %.2f RMSE' % (testScore))

# 绘制预测结果图
train_predict = []
for i in range(len(trainPredict)):
    train_predict.append(trainPredict[i][0])
np.array(train_predict)
# print(train_predict)
print(np.shape(train_predict))
trainPredictPlot = np.empty_like(low_f)
trainPredictPlot[:] = np.nan
trainPredictPlot[time_steps:len(trainPredict) + time_steps] = train_predict


test_predict = []
for i in range(len(testPredict)):
    test_predict.append(testPredict[i][0])
np.array(test_predict)
testPredictPlot = np.empty_like(low_f)
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict) + (time_steps * 2)-1:len(low_f) - 1] = test_predict


plt.plot(low_f, color='green')
plt.plot(trainPredictPlot, color='red')
plt.plot(testPredictPlot, color='blue')
plt.show()

# 变量保存函数
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename
# 保存变量
# filename_1 = save_variable(train_predict, r"C:\Users\31269\Desktop\毕设\variable\train_predict.txt")
filename_2 = save_variable(test_predict, r"C:\Users\31269\Desktop\毕设\variable\GRU_test_predict.txt")
