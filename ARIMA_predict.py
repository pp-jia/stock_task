import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from math import ceil
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels
import statsmodels.api as sm
import matplotlib.pyplot as plt

#  导入开盘价、最高价、最低价、换手率、成交量等数据
with open(r"C:\Users\31269\Desktop\毕设\data\沪深300期货当月.csv", 'rt') as csvfile:
    reader = csv.reader(csvfile)
    close_price = [row[1] for row in reader]

# 数据标准化
Stand = StandardScaler()
def normalize(data):
    data = np.array(data[1:len(data)])
    data = np.array(data)
    data = Stand.fit_transform(data.reshape(-1, 1)).T
    return data
close_price_normal = normalize(close_price)
print(np.shape(close_price_normal[0]))

# 平稳性检验  此函数给出了p值以及显著性水平，p值小于显著性水平—检验通过；如果检验不通过则进行差分，直到通过检验
def adf_test(timeseries):  # 传入时间序列
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
# 在这里进行平稳性检验
# adf_test(close_price_normal[0])
# Results of Dickey-Fuller Test:
# Test Statistic                     -1.416285
# p-value                             0.574444
# #Lags Used                         15.000000
# Number of Observations Used    175918.000000
# Critical Value (1%)                -3.430387
# Critical Value (5%)                -2.861556
# Critical Value (10%)               -2.566779
# dtype: float64

# 纯随机性检验（白噪声检验）：
data = sm.datasets.sunspots.load_pandas().data
res = statsmodels.tsa.arima.model.ARMA(data["SUNACTIVITY"], (1, 1)).fit(disp=-1)
sm.stats.acorr_ljungbox(res.resid, lags=[i for i in range(0, 10)], return_df=True)



# Read from csv file
df = pd.read_csv("AAPL_974.csv",header=0, index_col=0, parse_dates=True)
df = df.fillna(0)

# drop columns we don't want to predict
df.drop(df.columns[range(len(df.columns)-5,len(df.columns),1)], axis=1, inplace=True)

# split data
data_all_values = df.values
data_train_split = ceil(len(data_all_values)*0.80)
data_train = data_all_values[:data_train_split, :]
data_test = data_all_values[data_train_split:, :]

history = [x for x in data_train]
predictions = list()
for t in range(len(data_test)):
    model = ARIMA(history, order=(5, 1, 0)) #(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = data_test[t]
    history.append(obs)


# calculate MAE & RMSE
print('ARIMA RMSE: %.3f' % sqrt(mean_squared_error(data_test, predictions)))
print('ARIMA MAE: %.3f' % mean_absolute_error(data_test, predictions))
print('ARIMA rsquare: %.3f' % r2_score(data_test, predictions))

plt.plot(data_test, label="Original")
plt.plot(predictions, label="Predicted")
plt.title("ARIMA(0,1,1) Output")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend(loc="upper left")
plt.show()