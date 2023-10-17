import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.model_selection import GridSearchCV

# set option ###
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# loading the data ###
data = yf.download("THYAO.IS", start="2022-08-01", end="2022-09-01")
df = data.copy()
df = df.reset_index()
df.head()
# feature ###
df["day"] = df["Date"].dt.day    # method 2 - df["day"] = df["Date"].astype(str).str.split("-").str[2]
df = df.drop("Date", axis=1)

# variables ###
y = df["Adj Close"]
X = df["day"]

y = np.array(y).reshape(-1, 1)
X = np.array(X).reshape(-1, 1)

# train - test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standartiasion ###
scx = StandardScaler()
scy = StandardScaler()

X_train = scx.fit_transform(X_train)
y_train = scy.fit_transform(y_train)

X_test = scx.transform(X_test)

# modell ###
svrmodel = SVR(kernel="rbf")        # C = hiperparametresini ayarlayip hatayi minimize edebiliriz
svrmodel.fit(X_train, y_train)         # kernel   , gamma

# predict for trainset and testset
y_pred_train = svrmodel.predict(X_train)
y_pred_test = svrmodel.predict(X_test)

# r2 and rmse for trainset ###
r2_train = mt.r2_score(y_train, y_pred_train)               # 0.9593507524409128
rmse_train = mt.mean_squared_error(y_train, y_pred_train)   #  0.040649247559087116

# r2 and rmse for testset ###
r2_test = mt.r2_score(y_test, y_pred_test)                 # 0.8977119863439317
rmse_test = mt.mean_squared_error(y_test, y_pred_test)     #0.1429131795175042

# hiperparametre ###
parameters = {"C": [1, 500, 1000, 10000],
              "gamma": [1, 0.1, 0.001],
              "kernel": ["rbf", "linear", "poly"]
              }

tuning = GridSearchCV(estimator=SVR(), cv=17, param_grid=parameters)
tuning.fit(X_train, y_train)
tuning.best_params_         # {'C': 1, 'gamma': 1, 'kernel': 'rbf'}

# model with best params ###
svrmodel = SVR(**tuning.best_params_)
svrmodel.fit(X_train, y_train)

y_pred_train_h = svrmodel.predict(X_train)
y_pred_test_h = svrmodel.predict(X_test)

# r2 and rmse with best params ###
r2_train_h = mt.r2_score(y_train, y_pred_train_h)
rmse_train_h = mt.mean_squared_error(y_train, y_pred_train_h)


r2_test_h = mt.r2_score(y_test, y_pred_test_h)
rmse_test_h = mt.mean_squared_error(y_test, y_pred_test_h)

# grafik ###

plt.scatter(X_train, y_train, color="red", label="Gerçek Veriler")
plt.plot(X_train, y_pred_train, label="Tahminler", color="blue")
plt.xlabel("Gün")
plt.ylabel("Düzeltilmiş Kapanış Fiyatı")
plt.title("SVR Model Tahminleri")
plt.legend()
plt.show()
