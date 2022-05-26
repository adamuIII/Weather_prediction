import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def predict_weather():
    weather = pd.read_csv("local_weather.csv", index_col="DATE")

    core_weather = weather[["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]].copy()
    core_weather.columns = ["precip", "snow", "snow_depth", "temp_max", "temp_min"]


    weather.apply(pd.isnull).sum()/weather.shape[0]
    core_weather.apply(pd.isnull).sum()
    core_weather["snow"].value_counts()
    core_weather["snow_depth"].value_counts()
    del core_weather["snow"]
    del core_weather["snow_depth"]
    core_weather[pd.isnull(core_weather["precip"])]
    core_weather.loc["2013-12-15",:]
    core_weather["precip"].value_counts() / core_weather.shape[0]
    core_weather["precip"] = core_weather["precip"].fillna(0)
    core_weather.apply(pd.isnull).sum()
    core_weather[pd.isnull(core_weather["temp_min"])]
    core_weather.loc["2011-12-18":"2011-12-28"]
    core_weather = core_weather.fillna(method="ffill")
    core_weather.apply(pd.isnull).sum()
    core_weather.apply(lambda x: (x == 9999).sum())
    core_weather.dtypes
    core_weather.index
    core_weather.index = pd.to_datetime(core_weather.index)
    core_weather.index
    core_weather.index.year

    core_weather["temp_max"] = round(5/9 * (core_weather["temp_max"] - 32), 1)
    core_weather["temp_min"] = round(5/9 * (core_weather["temp_min"] - 32), 1)



    #porównanie temperatur na przestrzeni lat min i max
    fig = core_weather[["temp_max", "temp_min"]].plot().get_figure()
    fig.savefig("comparasion.png")


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
    fig = combined.plot().get_figure()
    fig.savefig('predictions.png')

    #wplyw poszczegolnych danych na temperature
    reg.coef_

if __name__ == "__main__":
    predict_weather()