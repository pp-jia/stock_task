import tushare as ts
import pandas as pd
import numpy as np
import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

sns.set_style("whitegrid", {"font.sans-serif": ['KaiTi', 'Arial']})


def run_main():
    k = ts.get_hist_data('600519')  # 600519茅台股票  这里可以设置获取的时间段
    # k  = ts.get_hist_data('600519',start='2015-05-04',end='2018-05-02')

    lit = ['open', 'high', 'close', 'low']  # 这里我们只获取其中四列
    data = k[lit]

    d_one = data.index  # 以下9行将object的index转换为datetime类型
    d_two = []
    d_three = []
    date2 = []
    for i in d_one:
        d_two.append(i)
    for i in range(len(d_two)):
        d_three.append(parse(d_two[i]))
    data2 = pd.DataFrame(data, index=d_three,
                         dtype=np.float64)  # 构建新的DataFrame赋予index为转换的d_three。当然你也可以使用date_range()来生成时间index

    plt.plot(data2['close'])  # 一看数据就不稳定，所以我们需要做差分
    plt.title('股市每日收盘价')
    plt.show()

    data2_w = data2['close'].resample('W-MON').mean()  # 由于原始数据太多，按照每一周来采样，更好预测，并取每一周的均值
    data2_train = data2_w['2015':'2017']  # 我们只取2015到2017的数据来训练
    plt.plot(data2_train)
    plt.title('周重采样数据')
    plt.show()

    # 一阶差分，分析ACF
    acf = plot_acf(data2_train, lags=20)  # 通过plot_acf来查看训练数据，以便我们判断q的取值
    plt.title("股票指数的 ACF")
    acf.show()

    # 一阶差分，分析PACF
    pacf = plot_pacf(data2_train, lags=20)  # 通过plot_pacf来查看训练数据，以便我们判断p的取值
    plt.title("股票指数的 PACF")
    pacf.show()

    # 处理数据，平稳化处理
    data2_diff = data2_train.diff(1)  # 差分很简单使用pandas的diff()函数可以进行一阶差分
    diff = data2_diff.dropna()
    for i in range(4):  # 五阶差分，一般一到二阶就行了，我有点过分
        diff = diff.diff(1)
        diff = diff.dropna()
    plt.figure()
    plt.plot(diff)
    plt.title('五阶差分')
    plt.show()

    # 五阶差分的ACF
    acf_diff = plot_acf(diff, lags=20)
    plt.title("五阶差分的ACF")  # 根据ACF图，观察来判断q
    acf_diff.show()

    # 五阶差分的PACF
    pacf_diff = plot_pacf(diff, lags=20)  # 根据PACF图，观察来判断p
    plt.title("五阶差分的PACF")
    pacf_diff.show()

    # 根据ACF和PACF以及差分 定阶并建模
    model = ARIMA(data2_train, order=(6, 1, 5), freq='W-MON')  # pdq    频率按周

    # 拟合模型
    arima_result = model.fit()

    # 预测
    pred_vals = arima_result.predict('2017-01-02', dynamic=True, typ='levels')  # 输入预测参数，这里我们预测2017-01-02以后的数据

    # 可视化预测
    stock_forcast = pd.concat([data2_w, pred_vals], axis=1, keys=['original', 'predicted'])  # 将原始数据和预测数据相结合，使用keys来分层

    # 构图
    plt.figure()
    plt.plot(stock_forcast)
    plt.title('真实值vs预测值')
    plt.show()


if __name__ == "__main__":
    run_main()