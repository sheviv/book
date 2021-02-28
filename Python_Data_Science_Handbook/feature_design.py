# Проектирование признаков
import numpy as np
import matplotlib.pyplot as plt

# прямое кодирование (one-hot encoding) - создание дополнительных столбцов-индикаторов наличия/отсутствия
# категории с помощью значений 1 или 0
# data = [
# {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
# {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
# {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
# {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
# ]
# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer(sparse=False, dtype=int)
# vec.fit_transform(data)
# # посмотреть названия признаков
# vec.get_feature_names()
# # разреженный формат вывода
# vec = DictVectorizer(sparse=True, dtype=int)
# vec.fit_transform(data)

# Текстовые признаки
# требуется преобразовывать текст в набор репрезентативных числовых значений
# Один из методов кодирования данных — по количеству слов
# sample = ['problem of evil','evil queen','horizon problem']
from sklearn.feature_extraction.text import CountVectorizer
# vec = CountVectorizer()
# X = vec.fit_transform(sample)  # количество вхождений каждого из слов
import pandas as pd
# cv = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
# «терма-обратная частотность документа» - слова получают вес с учетом частоты их появления во всех документах
from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer()
# X = vec.fit_transform(sample)
# cv = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())

# Производные признаки
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([4, 2, 1, 3, 7])
# X = x[:, np.newaxis]
# poly = PolynomialFeatures(degree=3, include_bias=False)
# X2 = poly.fit_transform(X)
# model = LinearRegression().fit(X2, y)
# yfit = model.predict(X2)
# plt.scatter(x, y)
# plt.plot(x, yfit)
# plt.show()

# Внесение отсутствующих данных
# from numpy import nan
# X = np.array([[nan, 0, 3],
#               [3, 7, 9],
#               [3, 5, 2],
#               [4, nan, 6],
#               [8, 8, 1]])
# y = np.array([14, 16, -1, 8, -5])
# заполнить(заменить) отсутствующие данные каким-либо подходящим значением
# замены пропущенных значений средним значением по столбцу
from sklearn.impute import SimpleImputer
# imp = SimpleImputer(strategy='mean')
# X2 = imp.fit_transform(X)
# можно передать, в LinearRegression
# model = LinearRegression().fit(X2, y)
# cv = model.predict(X2)

# Конвейеры признаков
# пример конвейера обработки:
#     1. Внести вместо отсутствующих данных средние значения.
#     2. Преобразовать признаки в квадратичные.
#     3. Обучить модель линейной регрессии.
# сборка конвейера
# from sklearn.pipeline import make_pipeline
# model = make_pipeline(SimpleImputer(strategy='mean'),PolynomialFeatures(degree=2),LinearRegression())
# model.fit(X, y) # Вышеприведенный массив X с пропущенными значениями
# print(y)
# print(model.predict(X))