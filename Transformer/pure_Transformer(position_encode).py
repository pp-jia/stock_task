import pickle
import sys

from sklearn.preprocessing import StandardScaler
sys.path.append("../../")
import csv
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torch.autograd import Variable
from Transformer_3 import TransformerTS

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# batch size
batch_size_train = 128
out_put_size = 1
seq_len = 32

# total epoch(总共训练多少轮)
total_epoch = 100

# . 导入训练数据
# 数据集导入
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

# 根据原始数据集构建矩阵
def create_dataset(data, time_steps, true_y):
    dataX, dataY = [], []
    # print("此时的inputshape：{}".format(len(data[1])))
    for i in range(len(data[0]) - time_steps):
        a = data[:, i:(i + time_steps)]
        dataX.append(a)
        dataY.append(true_y[0, i + time_steps])
    return np.array(dataX), np.array(dataY)

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
# print(open_price)

# print(np.shape(open_price))
print(low_f.reshape(-1, 1).T)
# all_data = np.concatenate((low_f.reshape(-1, 1).T, open_price_normal, high_price_normal, low_price_normal, volume_normal, chg_normal), axis=0)
all_data = np.concatenate((low_f.reshape(-1, 1).T, open_price_normal, high_price_normal, low_price_normal), axis=0)
print(np.shape(all_data))

# 切割为训练集和测试集
# print(len(low_f))
train_size = int(len(low_f) * 0.7)
val_size = int(len(low_f) * 0.2)
test_size = int(len(low_f) * 0.1)

train, val, test = all_data[:, 0:train_size], all_data[:, train_size:train_size+val_size], all_data[:, train_size+val_size:len(low_f)]
# print(np.shape(close_price_normal))
y_train = close_price_normal[:, 0:train_size]
y_val = close_price_normal[:, train_size:train_size+val_size]
y_test = close_price_normal[:, train_size+val_size:len(low_f)]

# print(y_test)

X_train, y_train = create_dataset(train, seq_len, y_train)
X_val, y_val = create_dataset(val, seq_len, y_val)
X_test, y_test = create_dataset(test, seq_len, y_test)

# reshape输入模型数据的格式为：[samples, time steps, features]
print(np.shape(X_train))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2], 4))  # 修改数据输入时这里需要更改
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 4))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[2], 4))

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

dataset_train = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False, drop_last=True)


# 2. 构建模型，优化器
tf = TransformerTS(input_dim=4,
                   dec_seq_len=10,
                   out_seq_len=out_put_size,
                   d_model=128,  # 编码器/解码器输入中预期特性的数量
                   nhead=10,
                   num_encoder_layers=6,
                   num_decoder_layers=6,
                   dim_feedforward=128,
                   dropout=0.1,
                   activation='relu',
                   custom_encoder=None,
                   custom_decoder=None).to(device)

optimizer = torch.optim.Adam(tf.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()  # mean square error
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 31433357277  # 网络训练过程中最大的loss


# 3. 模型训练
def train_transformer(epoch):
    global total_loss
    mode = True
    tf.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和
    # print(train_loader)
    for idx, (data, target) in enumerate(train_loader):
        data_np = data.numpy()
        target_np = target.numpy()

        data_np = data_np.astype(np.float32)
        target_np = target_np.astype(np.float32)

        data_torch = Variable(torch.tensor(data_np))  # 多维输入
        target_torch = torch.tensor(target_np)

        prediction = tf(data_torch.to(device))  # torch.Size([batch size])
        loss = criterion(prediction, target_torch.to(device))  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients
        #scheduler.step()
        #print(scheduler.get_lr())

        # print(loss)
        loss_epoch += loss.item()  # 将每个batch的loss累加，直到所有数据都计算完毕
        if idx == len(train_loader) - 1:
            print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
            train_loss_list.append(loss_epoch)
            if loss_epoch < total_loss:
                total_loss = loss_epoch
                torch.save(tf, 'tf_model2.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    for i in range(total_epoch):  # 模型训练100轮
        train_transformer(i)
    print("Stop Training!")