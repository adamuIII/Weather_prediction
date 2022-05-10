import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

weather = pd.read_csv("local_weather.csv", index_col="DATE")

core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]



#porównanie temperatur na przestrzeni lat min i max
core_weather[["temp_max", "temp_min"]].plot()


#wyswietlenie opadow
core_weather["precip"].plot()


#tworzymy kolumne z przewidywana pogoda
#shift(-1) cofa nam max temp o jeden do targetu dzieki temu targetem jest jutrzejsza temperatura
#jesli we wtorek temp wynosi 30 to taki jest target w poniedzialek
core_weather["target"] = core_weather.shift(-1)["temp_max"]
#wyłączamy ostatni bo nie ma z czego pobierać
core_weather = core_weather.iloc[:-1,:].copy()


#zaczynamy machine learning Regresja grzbietowa
reg = Ridge(alpha=.1)

#do przewidywania używamy opadów, temp max i temp min
predictors = ["precip", "temp_max", "temp_min"]
#dane uczace
train = core_weather.loc[:"2020-12-31"]
#dane testowe
test = core_weather.loc["2021-01-01":]
train

test


reg.fit(train[predictors], train["target"])
predictions = reg.predict(test[predictors])



#sredni blad bezwzgledny
mean_squared_error(test["target"], predictions)

#predykcje
combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]
combined

#predykcje na grafie
combined.plot()


#wplyw poszczegolnych danych na temperature
reg.coef_