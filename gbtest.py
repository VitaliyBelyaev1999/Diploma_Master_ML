import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


reg = joblib.load("D:\StudyProgramming\Python\MDDA\GragientBoostingApproximator.joblib.dat")
df = pd.read_csv("gbtest.csv", sep = " ", index_col=False)
df = df.replace({',': '.'}, regex=True).astype(float)
y_pred1 = reg.predict(df)
np.savetxt('ypredicted.csv', y_pred1, delimiter=' ')