import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model

weatherHistory = pd.read_csv("data\weatherHistory.csv")
print(weatherHistory.head())
print(weatherHistory.describe())


weatherFeatures = ["Temperature (C)","Apparent Temperature (C)","Wind Speed (km/h)", 
                   "Wind Bearing (degrees)","Visibility (km)","Pressure (millibars)"]

x = weatherHistory[weatherFeatures]
y = weatherHistory.Humidity

x_scaled = preprocessing.scale(x)
poly = PolynomialFeatures(1)

x_final = poly.fit_transform(x_scaled)

x_train,x_test,y_train,y_test = train_test_split(x_final,y,test_size=0.10,random_state = 42)


regr = linear_model.Ridge(alpha = 0.5)
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

print(y_pred)
