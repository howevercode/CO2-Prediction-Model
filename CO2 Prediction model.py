import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv("data.csv")

print(df.head)

x = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(x, y)

predictedCO2 = regr.predict([[1300, 800]])

print(predictedCO2)
