from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
from math import ceil
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

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