# Линейная регрессия

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

# Простая линейная регрессия
# rng = np.random.RandomState(1)
# x = 10 * rng.rand(50)
# y = 2 * x - 5 + rng.randn(50)
# plt.scatter(x, y)
# plt.show()
# обучения на этих данных и поиска оптимальной прямой
# coef_ и intercept_ - включают угловой коэффициент и точку пересечения с осью координат
from sklearn.linear_model import LinearRegression
# model = LinearRegression(fit_intercept=True)
# model.fit(x[:, np.newaxis], y)
# xfit = np.linspace(0, 10, 1000)
# yfit = model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()
# одна из этих аппроксимаций в действии, создав данные для нашего примера с помощью оператора матричного умножения
# rng = np.random.RandomState(1)
# X = 10 * rng.rand(100, 3)
# y = 0.5 + np.dot(X, [1.5, -2., 1.])
# model.fit(X, y)

# Полиномиальные базисные функции
from sklearn.preprocessing import PolynomialFeatures
# x = np.array([2, 3, 4])
# poly = PolynomialFeatures(3, include_bias=False)
# poly.fit_transform(x[:, None])
# преобразователь превратил одномерный массив в трехмерный путем возведения каждого из значений в степень
# конвейер для «Проектирование признаков»:
from sklearn.pipeline import make_pipeline
# poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
# можно воспользоваться линейной моделью для подбора намного более сложных зависимостей между величинами x и y
# rng = np.random.RandomState(1)
# xfit = np.linspace(0, 10, 1000)
# x = 10 * rng.rand(50)
# y = np.sin(x) + 0.1 * rng.randn(50)
# poly_model.fit(x[:, np.newaxis], y)
# yfit = poly_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()
# сумма не полиномиальных, а Гауссовых базисных функций
# Затененные области — нормированные базисные функции, дающие при сложении аппроксимирующую данные гладкую кривую
# from sklearn.base import BaseEstimator, TransformerMixin
# class GaussianFeatures(BaseEstimator, TransformerMixin):
#     """Равномерно распределенные Гауссовы признаки для одномерных входных данных"""
#     def __init__(self, N, width_factor=2.0):
#         self.N = N
#         self.width_factor = width_factor
#     @staticmethod
#     def _gauss_basis(x, y, width, axis=None):
#         arg = (x - y) / width
#         return np.exp(-0.5 * np.sum(arg ** 2, axis))
#     def fit(self, X, y=None):
#         # Создаем N центров, распределенных по всему диапазону данных
#         self.centers_ = np.linspace(X.min(), X.max(), self.N)
#         self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
#         return self
#     def transform(self, X):
#         return self._gauss_basis(X[:, :, np.newaxis], self.centers_, self.width_, axis=1)
# gauss_model = make_pipeline(GaussianFeatures(20), LinearRegression())
# gauss_model.fit(x[:, np.newaxis], y)
# yfit = gauss_model.predict(xfit[:, np.newaxis])
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.xlim(0, 10)
# plt.show()

# Регуляризация
# model = make_pipeline(GaussianFeatures(30),LinearRegression())
# model.fit(x[:, np.newaxis], y)
# plt.scatter(x, y)
# plt.plot(xfit, model.predict(xfit[:, np.newaxis]))
# plt.xlim(0, 10)
# plt.ylim(-1.5, 1.5)
# plt.show()
# В результате проекции данных на 30-мерный базис модель оказалась слишком уж гибкой и стремится к экстремальным значениям
# в промежутках между точками, в которых она ограничена данными
# график коэффициентов Гауссовых базисных функций в соответствии с координатой x
# def basis_plot(model, title=None):
#     fig, ax = plt.subplots(2, sharex=True)
#     model.fit(x[:, np.newaxis], y)
#     ax[0].scatter(x, y)
#     ax[0].plot(xfit, model.predict(xfit[:, np.newaxis]))
#     ax[0].set(xlabel='x', ylabel='y', ylim=(-1.5, 1.5))
#     if title:
#         ax[0].set_title(title)
#     ax[1].plot(model.steps[0][1].centers_,
#                model.steps[1][1].coef_)
#     ax[1].set(xlabel='basis location',  # Базовое местоположение
#               ylabel='coefficient',  # Коэффициент
#               xlim=(0, 10))
#     plt.show()
# model = make_pipeline(GaussianFeatures(30), LinearRegression())
# basis_plot(model)

# Гребневая регрессия(L2-регуляризация| регуляризацией Тихонова) - наложение штрафа на сумму квадратов(евклидовой нормы)
# коэффициентов модели.
from sklearn.linear_model import Ridge
# model = make_pipeline(GaussianFeatures(30), Ridge(alpha=0.1))
# basis_plot(model, title='Ridge Regression') # Гребневая регрессия

# Лассо-регуляризация(L1) - штрафование на сумму абсолютных значений(L1-норма) коэффициентов регрессии(по возможности
# делает коэффициенты модели равными нулю)
from sklearn.linear_model import Lasso
# model = make_pipeline(GaussianFeatures(30), Lasso(alpha=0.001))
# basis_plot(model, title='Lasso Regression') # Лассо-регуляризация

# Пример: предсказание велосипедного трафика
# Начнем с загрузки двух наборов данных, индексированных по дате:
# import pandas as pd
# counts = pd.read_csv('fremont_hourly.csv', index_col='Date', parse_dates=True)
# weather = pd.read_csv('weather.csv', index_col='DATE', parse_dates=True)
# # вычислим общий ежедневный поток велосипедов
# # daily = counts.resample('d', how='sum')
# daily = counts.resample('d').sum()
# daily['Total'] = daily.sum(axis=1)
# daily = daily[['Total']]  # удаляем остальные столбцы
# # паттерны использования варьируются день ото дня. добавив двоичные столбцы-индикаторы дня недели:
# days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
# for i in range(7):
#     daily[days[i]] = (daily.index.dayofweek == i).astype(float)
# # Следует ожидать, что велосипедисты будут вести себя иначе по выходным. Добавим индикаторы
# from pandas.tseries.holiday import USFederalHolidayCalendar
# cal = USFederalHolidayCalendar()
# holidays = cal.holidays('2012', '2016')
# daily = daily.join(pd.Series(1, index=holidays, name='holiday'))
# daily['holiday'].fillna(0, inplace=True)
# # световой день - воспользуемся стандартными астрономическими расчетами
# def hours_of_daylight(date, axis=23.44, latitude=47.61):
#     """Рассчитываем длительность светового дня для заданной даты"""
#     days = (date - pd.datetime(2000, 12, 21)).days
#     m = (1. - np.tan(np.radians(latitude)) * np.tan(np.radians(axis) * np.cos(days * 2 * np.pi / 365.25)))
#     return 24. * np.degrees(np.arccos(1 - np.clip(m, 0, 2))) / 180.
# daily['daylight_hrs'] = list(map(hours_of_daylight, daily.index))
# daily[['daylight_hrs']].plot()
# plt.show()
# добавим к данным среднюю температуру и общее количество осадков
# добавим еще и флаг для обозначения засушливых дней (с нулевым количеством осадков)
# Температуры указаны в десятых долях градуса Цельсия преобразуем в градусы
# weather['TMIN'] /= 10
# weather['TMAX'] /= 10
# weather['Temp (C)'] = 0.5 * (weather['TMIN'] + weather['TMAX'])
# # Осадки указаны в десятых долях миллиметра; преобразуем в дюймы
# weather['PRCP'] /= 254
# weather['dry day'] = (weather['PRCP'] == 0).astype(int)
# daily = daily.join(weather[['PRCP', 'Temp (C)', 'dry day']])