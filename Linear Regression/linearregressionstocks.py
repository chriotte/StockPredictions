import quandl as Quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import csv

style.use('ggplot')

stock = 'EIA/PET_RWTC_D'
quandlApi_key = "CzD8D1mR6eebi9sNnqpc"

CSV_URL = ("https://www.quandl.com/api/v3/datasets/%s.csv"% (stock)) + "?" + quandlApi_key  
rawData = pd.read_csv(CSV_URL, quoting=csv.QUOTE_NONE, error_bad_lines=False, parse_dates=True, index_col=[0])
          
df = rawData
forecast_col = 'Value'

df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))

X = preprocessing.scale(X)

X_lately = X[-forecast_out:]

X = X[:-forecast_out]

df.dropna(inplace=True)
df.sort_index(inplace=True)


y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
forecast_set2 = np.flipud(forecast_set) 

for i in forecast_set2:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]


df['Value'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

