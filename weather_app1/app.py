from flask import Flask, render_template, request
import requests
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Ваш API-ключ OpenWeatherMap
API_KEY = 'f2718556e86a87a5e72ce54824bc6aaa'
URL = 'http://api.openweathermap.org/data/2.5/onecall/timemachine'


def get_city_coordinates(city_name):
    """Получаем координаты города по его названию."""
    geocoding_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={API_KEY}"
    response = requests.get(geocoding_url)
    data = response.json()
    if not data:
        raise ValueError(f"Город '{city_name}' не найден. Пожалуйста, проверьте название.")
    return data[0]['lat'], data[0]['lon']


def get_historical_weather(lat, lon, year, month):
    """Получаем исторические данные о погоде за конкретный месяц и год."""
    dt = int(datetime(year, month, 1).timestamp())
    params = {
        'lat': lat,
        'lon': lon,
        'dt': dt,
        'appid': API_KEY,
        'units': 'metric'
    }
    response = requests.get(URL, params=params)
    if response.status_code != 200:
        print(f"Ошибка: {response.json()}")
        return None
    return response.json()


def parse_weather_data(data):
    """Извлекаем среднюю температуру из данных API."""
    temps = [day['temp'] for day in data['hourly']]
    avg_temp = sum(temps) / len(temps)
    return avg_temp


def collect_historical_data(lat, lon, start_year, end_year):
    """Собираем исторические данные за указанный период."""
    historical_data = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            data = get_historical_weather(lat, lon, year, month)
            if data and 'hourly' in data:
                avg_temp = parse_weather_data(data)
                historical_data.append({'year': year, 'month': month, 'avg_temp': avg_temp})
            else:
                print(f"Данные отсутствуют для {year}-{month}.")
    df = pd.DataFrame(historical_data)
    if df.empty:
        print("Исторические данные отсутствуют.")
    else:
        print("Исторические данные:")
        print(df)
    return df


def predict_future_temperatures(df, future_year):
    """Прогнозируем температуру на будущий год с помощью линейной регрессии."""
    # Удаляем строки с NaN
    df = df.dropna()
    if df.empty:
        raise ValueError("Нет данных для обучения модели.")

    # Заполняем пропущенные значения средним
    df['avg_temp'] = df['avg_temp'].fillna(df['avg_temp'].mean())

    X = df[['year', 'month']]
    y = df['avg_temp']
    model = LinearRegression()
    model.fit(X, y)
    print(f"Коэффициенты модели: intercept={model.intercept_}, coef={model.coef_}")

    future_data = []
    for month in range(1, 13):
        future_data.append({'year': future_year, 'month': month})
    future_df = pd.DataFrame(future_data)
    future_df['predicted_temp'] = model.predict(future_df[['year', 'month']])
    return future_df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city_name = request.form['city']
        try:
            lat, lon = get_city_coordinates(city_name)
            historical_df = collect_historical_data(lat, lon, 2014, 2023)
            if historical_df.empty:
                return render_template('index.html', error="Нет исторических данных для этого города.")
            future_df = predict_future_temperatures(historical_df, 2025)
            predictions = future_df[['month', 'predicted_temp']].to_dict('records')
            return render_template('index.html', city=city_name, predictions=predictions)
        except ValueError as e:
            error = str(e)
            return render_template('index.html', error=error)
        except Exception as e:
            error = f"Произошла ошибка: {str(e)}"
            return render_template('index.html', error=error)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)