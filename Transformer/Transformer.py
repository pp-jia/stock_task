#!/usr/bin/env Python
# coding=utf-8
# 使用transformer进行沪深300指数收盘价预测
# 导入相关包
import pickle
import numpy as np
import csv
from keras.layers import LayerNormalization
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
import matplotlib.pyplot as plt

# using colorblind palette

plt.style.use('seaborn-v0_8-colorblind')
import warnings

warnings.filterwarnings('ignore')
print('Tensorflow version: {}'.format(tf.__version__))

# 超参数
batch_size = 256
seq_len = 32
num_epochs = 8
d_k = 256
d_v = 256
n_heads = 12
ff_dim = 256

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
    # print("此时的input_shape：{}".format(len(data[1])))
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
all_data = np.concatenate((low_f.reshape(-1, 1).T, open_price_normal, high_price_normal, low_price_normal), axis=0)
# all_data = low_f.reshape(-1, 1).T
print(np.shape(all_data))

# 切割为训练集和测试集
# print(len(low_f))
train_size = int(len(low_f) * 0.7)
val_size = int(len(low_f) * 0.2)
test_size = int(len(low_f) * 0.1)

train, val, test = all_data[:, 0:train_size], all_data[:, train_size:train_size + val_size], all_data[:,
                                                                                             train_size + val_size:len(
                                                                                                 low_f)]
# print(np.shape(close_price_normal))
y_train = close_price_normal[:, 0:train_size]
y_val = close_price_normal[:, train_size:train_size + val_size]
y_test = close_price_normal[:, train_size + val_size:len(low_f)]

# print(y_test)

X_train, y_train = create_dataset(train, seq_len, y_train)
X_val, y_val = create_dataset(val, seq_len, y_val)
X_test, y_test = create_dataset(test, seq_len, y_test)

# reshape输入模型数据的格式为：[samples, time steps, features]
print(np.shape(X_train))
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[2], 4))  # 修改数据输入时这里需要更改
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[2], 4))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[2], 4))


# Time2Vector
class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        '''Initialize weights and biases with shape (batch, seq_len)'''
        self.weights_linear = self.add_weight(name='weight_linear',
                                              shape=(int(self.seq_len),),
                                              initializer='uniform',
                                              trainable=True)

        self.bias_linear = self.add_weight(name='bias_linear',
                                           shape=(int(self.seq_len),),
                                           initializer='uniform',
                                           trainable=True)

        self.weights_periodic = self.add_weight(name='weight_periodic',
                                                shape=(int(self.seq_len),),
                                                initializer='uniform',
                                                trainable=True)

        self.bias_periodic = self.add_weight(name='bias_periodic',
                                             shape=(int(self.seq_len),),
                                             initializer='uniform',
                                             trainable=True)

    def call(self, x):
        '''Calculate linear and periodic time features'''
        x = tf.math.reduce_mean(x[:, :, :4], axis=-1)
        time_linear = self.weights_linear * x + self.bias_linear  # Linear time feature
        time_linear = tf.expand_dims(time_linear, axis=-1)  # Add dimension (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis=-1)  # Add dimension (batch, seq_len, 1)
        return tf.concat([time_linear, time_periodic], axis=-1)  # shape = (batch, seq_len, 2)

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'seq_len': self.seq_len})
        return config


# position enciding
class PositionalEncoding(Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)
        # 将 sin 应用于数组中的偶数索引（indices）；2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # 将 cos 应用于数组中的奇数索引；2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)  # Casts a tensor to a new type.

    def build(self, input_shape):
        self.pos_encoding = self.positional_encoding(input_shape[1], input_shape[2])

    def call(self, inputs):
        return inputs + self.pos_encoding


# Transformer
class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

        self.key = Dense(self.d_k,
                         input_shape=input_shape,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='glorot_uniform')

        self.value = Dense(self.d_v,
                           input_shape=input_shape,
                           kernel_initializer='glorot_uniform',
                           bias_initializer='glorot_uniform')

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b=True)
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis=-1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out

    #############################################################################


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads):
            self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
            # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
        self.linear = Dense(input_shape[0][-1],
                            input_shape=input_shape,
                            kernel_initializer='glorot_uniform',
                            bias_initializer='glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis=-1)
        multi_linear = self.linear(concat_attn)
        return multi_linear
    #############################################################################


class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout=0.1, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.attn_heads = list()
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

        self.ff_conv1D_1 = Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
        # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1] = 7
        self.ff_conv1D_2 = Conv1D(filters=input_shape[0][-1], kernel_size=1)
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)

        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

    def get_config(self):  # Needed for saving and loading model with custom layer
        config = super().get_config().copy()
        config.update({'d_k': self.d_k,
                       'd_v': self.d_v,
                       'n_heads': self.n_heads,
                       'ff_dim': self.ff_dim,
                       'attn_heads': self.attn_heads,
                       'dropout_rate': self.dropout_rate})
        return config


# 模型构建
def create_model():
    '''Initialize time and transformer layers'''
    # time_embedding = Time2Vector(seq_len)  # 使用了Time2Vector

    position_encoding = PositionalEncoding()  # 使用了position_encoding
    attn_layer1 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    # attn_layer2 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)
    # attn_layer3 = TransformerEncoder(d_k, d_v, n_heads, ff_dim)

    '''Construct model'''
    in_seq = Input(shape=(seq_len, 4))
    # x = time_embedding(in_seq)
    # x = Concatenate(axis=-1)([in_seq, x])

    x = position_encoding(in_seq)
    x = attn_layer1((x, x, x))
    # x = attn_layer2((x, x, x))
    # x = attn_layer3((x, x, x))
    # x = GlobalAveragePooling1D(data_format='channels_first')(x)
    # x = Dropout(0.1)(x)

    # decoder 部分用LSTM替代
    x = tf.reshape(x, [-1, seq_len, 4])  # LSTM的3-D输入-->[batch_size, seq_length, input_dim]
    x = LSTM(128, input_shape=(seq_len, 4))(x)
    x = Dropout(0.1)(x)
    # x = LSTM(128, return_sequences=False)(x)
    # x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    out = Dense(1)(x)

    model = Model(inputs=in_seq, outputs=out)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae', 'mape'])
    return model


model = create_model()
model.summary()

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=num_epochs,
                    # steps_per_epoch=len(X_train)/batch_size,
                    validation_data=(X_val, y_val))

###############################################################################
'''Calculate predictions and metrics'''

# Calculate predication for training, validation and test data
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)


# 变量保存函数
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


# 保存变量
filename_1 = save_variable(test_pred, r"C:\Users\31269\Desktop\毕设\variable\transformer_test_pred.txt")

# loss曲线绘制
def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
visualize_loss(history, "Loss")


# Print evaluation metrics for all datasets
train_eval = model.evaluate(X_train, y_train, verbose=0)
val_eval = model.evaluate(X_val, y_val, verbose=0)
test_eval = model.evaluate(X_test, y_test, verbose=0)
print(' ')
print('Evaluation metrics')
print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

###############################################################################
