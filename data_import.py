import pandas as pd
from WindPy import w
w.start()

Wind_Data = w.wsi("USDCNY.IB", "close", "2023-01-15 09:00:00", "2023-01-15 18:39:24", "")

# print(Wind_Data[1])
print(Wind_Data.Data[0][0])
print(len(Wind_Data.Data[0]))

print(Wind_Data.Times[0])
# df = pd.DataFrame(Wind_Data.Data, columns=Wind_Data.Times, index=["close","open","high","low","volume","amt","chg", "pct_chg","oi","BIAS","BOLL","KDJ","MA","MACD","RSI"]).T

df = pd.DataFrame(Wind_Data.Data, columns=Wind_Data.Times, index=["close"]).T
df.to_csv('C:\\Users\\31269\\Desktop\\毕设\\data\\美元兑人名币.csv', index=True)


