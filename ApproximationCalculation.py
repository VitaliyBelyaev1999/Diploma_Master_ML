import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor


def tfunc_reg_test1(x, a, b, c, d, e):
    
    # power(b) = x[0], mu1(c) = x[1], mu2(d) = x[2], sigma(e) = x[3], coef = x[4], alpha(a) = x[5]
    quantile1 = norm.ppf(a * x[5] / 2)
    quantile2 = norm.ppf(1 - b * x[0])
    #d1 = delta / (sqrt(gamma(3/x[4]) / gamma(1/x[4])) / (d * x[3]))
    d1 = (c * x[1] - d * x[2]) / (e * x[3])
    tfunc = ((quantile1 + quantile2) / d1) ** 2
    return tfunc


def LR_model(df): # Нелинейная регрессия
    
    df = df[df['alpha'] == '0.001']
    df = df[df['GND_shape_parameter'] == '1'] 
    
    y = df['n1'].astype(float).to_numpy()
    power = df['Power'].astype(float).to_numpy()
    mu1 = df['mu1'].astype(float).to_numpy()
    mu2 = df['mu2'].astype(float).to_numpy()
    sigma1 = df['sigma1'].astype(float).to_numpy()
    alpha = df['alpha'].astype(float).to_numpy()
    coef = df['GND_shape_parameter'].astype(float).to_numpy()
    x = np.vstack((power, mu1, mu2, sigma1, coef, alpha))

    p0 = [1.02, 0.99, 1.2, 1, 1.04]
    popt, pcov = curve_fit(tfunc_reg_test1, x, y, p0)
    print("Оптимальные параметры:", popt)  
    y1 = tfunc_reg_test1(x, *popt)
    print(r2_score(y, y1))
    plt.scatter(x[0], y, s=6, color='r', label='Исходные данные')
    plt.scatter(x[0], y1, s=1, color='b', label='Модель')
    plt.xlabel("Статистическая мощность")
    plt.ylabel("Объём выборки")
    plt.legend(loc="upper left")
    plt.show()
    return()


def ML(df): # Градиентный бустинг
    
    

    power = df['Power'].astype(float).to_numpy()
    X = df.drop('n1', axis = 1).astype(float)
    print(X)
    y = df['n1'].astype(float)

    power = power.reshape(-1, 1)
    #y = y.reshape(-1, 1)
    params = {
        "max_depth": 10,
        "learning_rate": 1,
        "criterion": 'mse',
    }
    reg = GradientBoostingRegressor(**params)
    reg.fit(X, y)
    y_pred = reg.predict(X)  
    print(r2_score(y, y_pred))

    #np.savetxt('ypredicted.csv', y_pred, delimiter=',')    

    joblib.dump(reg, "D:\StudyProgramming\Python\MDDA\GragientBoostingApproximator.joblib.dat")
    plt.scatter(
        df['Power'].astype(float),
        y,
        s=6,
        c = "r",
        label="original",
    )
    plt.scatter(
        df['Power'].astype(float), 
        y_pred, 
        s=1,
        c = "b", 
        label="prediction"
    )
    plt.xlabel("Статистическая мощность")
    plt.ylabel("Объём выборки")
    plt.legend(loc="upper left")
    plt.show()

    
    return()


df = pd.read_csv("fulldb.csv", sep = " ", index_col=False)
df = df[['Power', 'n1', 'n2', 'mu1', 'mu2', 'sigma1', 'GND_shape_parameter', 'alpha']]
df = df.replace({',': '.'}, regex=True)
df = df[df['n1'] == df['n2']]
df = df[['Power', 'n1', 'mu1', 'mu2', 'sigma1', 'GND_shape_parameter', 'alpha']]

LR_model(df)
#ML(df)