import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

weatherHistory = pd.read_csv("data\weatherHistory.csv")
print(weatherHistory.head())
print(weatherHistory.describe())


weatherFeatures = ["Humidity","ApparentTemperature(C)","WindSpeed(km/h)", 
                   "WindBearing(degrees)","Visibility(km)","Pressure(millibars)"]

x = weatherHistory[weatherFeatures]
y = weatherHistory.TemperatureC

x_scaled = preprocessing.scale(x)
poly = PolynomialFeatures(7)

x_final = poly.fit_transform(x_scaled)

x_train,x_test,y_train,y_test = train_test_split(x_final,y,test_size=0.10,random_state = 42)


regr = linear_model.Ridge(alpha = 0.5)
regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

print(y_pred)

print("mean-squared-error: %.3f"% mean_squared_error(y_test,y_pred))
print("coeffecient of determination: %.3f" % r2_score(y_test, y_pred))

print("intercept: ", regr.intercept_)
print("coeffecients: ", len(regr.coef_))

wilgotnosc =  float(input("Podaj wilgotnosc"))
tempOdczuwalna =  float(input("Podaj temp odczuwalna"))
wiatr =  float(input("Podaj predkosc wiatru"))
wiatrKierunek =  float(input("Podaj kierunek wiatru w stopniach"))
widocznosc = float(input("Podaj widocznosc w km"))
cisnienie = float(input("Podaj cisnienie"))

weatherObs = [[wilgotnosc,tempOdczuwalna,wiatr,wiatrKierunek,widocznosc,cisnienie]]
weatherObs_scaled = preprocessing.scale(weatherObs)
weatherObs_final = poly.fit_transform(weatherObs_scaled)

y_pred = regr.predict(weatherObs_final)
print(y_pred)