import datetime
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import datetime as dt
import pandas_datareader.data as web
import math
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Define the period for historical stock prices
start = dt.datetime(2017, 1, 1)
end = dt.datetime(2019, 9, 9)

# Use pandas data_reader to extract our Google Stocks prices from yahoo.
df = web.DataReader("GOOG", 'yahoo', start, end)
# Pulling some registers for checking it out.
df.tail()

# Predicting Stock Price through Simple Linear Analysis, Quadratic Discriminant Analysis (QDA),
# and K Nearest Neighbor (KNN)

# Define features: High Low Percentage and Percentage Change.
df_reg = df.loc[:, ['Adj Close', 'Volume']]
df_reg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
df_reg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

print(df_reg)

# Preparing our data, cross-validation https://www.textbook.ds100.org/ch/15/bias_cv.html:
# 1. Filling missing values
df_reg.fillna(df_reg.mean(), inplace=True)

# 2. Retrieve 1% of our data for forecasting.
forecast_out = int(math.ceil(0.01 * len(df_reg)))

# Separating the labels, we want to predict the Adj Close.
forecast_col = 'Adj CLose'
df_reg['label'] = df_reg['Adj Close'].shift(forecast_out)
X = np.array(df_reg.drop(['label'], 1))

# Scale X in order to have the same distribution for linear regression (Normalization).
X = sk.preprocessing.scale(X)

# Find Data Series of late X and early X (train) for model generation and evaluation.
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

# Separate label and identify it as y
y = np.array(df_reg['label'])
y = y[:-forecast_out]

# Separation of training and testing of model by cross validation train test split
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size=0.1, random_state=0)

# Filtering NaN values
X_train = X_train[~np.isnan(X_train)]
y_train = y_train[~np.isnan(y_train)]

# Reshape our training set dimension
X_train = pd.factorize(df_reg['label'])[0].reshape(-1, 1)
y_train = pd.factorize(df_reg['label'])[0].reshape(-1, 1)

# Model generation -> Simple Linear Analysis, Quadratic Discriminant Analysis and KNN regression
# Play the existing Scikit-Learn library and train the model by selecting our X and y train sets.

# Linear regression
clf_reg = sk.linear_model.LinearRegression(n_jobs=-1)
clf_reg.fit(X_train, y_train)


# Quadratic Regression 2
clf_poly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clf_poly2.fit(X_train, y_train)
clf_reg.reshape(-1, 1)


# Quadratic Regression 3
clf_poly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clf_poly3.fit(X_train, y_train)

# KNN Regression
clf_knn = KNeighborsRegressor(n_neighbors=2)
clf_knn.fit(X_train, y_train)

# Evaluating our models through the score method
confidence_reg = clf_reg.score(X_test, y_test)
confidence_poly2 = clf_poly2.score(X_test,y_test)
confidence_poly3 = clf_poly3.score(X_test,y_test)
confidence_knn = clf_knn.score(X_test, y_test)

# Some of the stocks forecast.
forecast_set = clf_reg.predict(X_lately)
df_reg['Forecast'] = np.nan


# Visualizing the plot with our existing historical data.
last_date = df_reg.iloc[-1].name
last_unix = last_date
next_unix = last_unix + datetime.timedelta(days=1)

for i in forecast_set:
    next_date = next_unix
    next_unix += datetime.timedelta(days=1)
    df_reg.loc[next_date] = [np.nan for _ in range(len(df_reg.columns)-1)]+[i]
df_reg['Adj Close'].tail(500).plot()
df_reg['Forecast'].tail(500).plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

















