import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Подготовка данных
data = {
    'year': [2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004],
    'oilprice': [71.06, 54.25, 43.55, 52.35, 99.03, 108.56, 111.63, 111.27, 79.47, 61.51, 96.99, 72.52, 65.14, 54.38, 38.10],
    'gdp': [1657.554647, 1578.624061, 1282.723881, 1363.594370, 2059.984158, 2297.128039, 2210.256977, 2051.661732, 1524.917468, 1222.644282, 1660.846388, 1299.705765, 989.930542, 764.017108, 591.016691]
}
df = pd.DataFrame(data)

# Разделение данных на признаки и целевую переменную
X = df[['oilprice']]
y = df['gdp']

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred_test = model.predict(X_test)

# Вычисление метрик
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)
print("Predicted values (y_pred_test):", y_pred_test)
print("Actual values (y_test):", y_test.values)

from sklearn.ensemble import RandomForestRegressor

# Обучение модели случайного леса
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred_test_rf = rf_model.predict(X_test)

# Вычисление метрик
mse_rf = mean_squared_error(y_test, y_pred_test_rf)
r2_rf = r2_score(y_test, y_pred_test_rf)

print("Random Forest - Mean Squared Error (MSE):", mse_rf)
print("Random Forest - R-squared (R²):", r2_rf)


from sklearn.ensemble import GradientBoostingRegressor

# Обучение модели градиентного бустинга
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred_test_gb = gb_model.predict(X_test)

# Вычисление метрик
mse_gb = mean_squared_error(y_test, y_pred_test_gb)
r2_gb = r2_score(y_test, y_pred_test_gb)

print("Gradient Boosting - Mean Squared Error (MSE):", mse_gb)
print("Gradient Boosting - R-squared (R²):", r2_gb)
