# ch_4_Работа с числовыми данными

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

# прошкалировать числовой признак в диапазон между двумя значениями
# Создать признак
xc= np.array([[-500.5], [-100.1], [0], [100.1], [900.9]])
# Создать шкалировщик
# minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
# Прошкалировать признак
# scaled_feature = minmax_scale.fit_transform(xc)

# преобразовать признак, чтобы он имел среднее значение 0 и стандартное отклонение 1
# Создать шкалировщик
# scaler = preprocessing.StandardScaler()
# Преобразовать признак
# standardized = scaler.fit_transform(xc)

# среднее значение и стандартное отклонение
# print("Среднее:", round(standardized.mean()))
# print("Стандартное отклонение:", standardized.std())

# метод робастного шкалирования
# Создать шкалировщик
# robust_scaler = preprocessing.RobustScaler()
# Преобразовать признак
# robust_scaler.fit_transform(xc)

# прошкалировать значения признаков в наблюдениях для получения единичной нормы (общей длиной 1)
# Создать нормализатор
# normaiizer = Normalizer(norm="l2")
# Преобразовать матрицу признаков
# normaiizer.transform(xc)

# евклидова норма (нередко именуемая L2 - нормой)
# Преобразовать матрицу признаков
# features_12_norm = Normalizer(norm="l2").transform(x)

# манхэттенскя норма (L') || шкалирует значения наблюдений таким образом, что они в сумме дают 1.
# Преобразовать матрицу признаков
# features_l1_norm = Normalizer(norm="l1").transform(xc)

# создать полиномиальные и взаимодействующие признаки
from sklearn.preprocessing import PolynomialFeatures
# Создать матрицу признаков
# xc = np.array([[2, 3], [2, 3], [2, 3]])
# Создать объект PolynomialFeatures
# polynomial_interaction = PolynomialFeatures(degree=2, include_bias=False)
# Создать полиномиальные признаки
# cv = polynomial_interaction.fit_transform(xc)

# только признаки взаимодействия, установив для interaction_only=True
# interaction = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
# interaction.fit_transform(xc)

# выполнить собственное преобразование одного или более признаков(вить ко всем признакам число)
from sklearn.preprocessing import FunctionTransformer
# Определить простую функцию
# def add_ten(x):
#     return x + 10
# Создать преобразователь
# ten_transformer = FunctionTransformer(add_ten)
# Преобразовать матрицу признаков
# ten transformer.transform(xc)

# обработка выбросов.
# 1. отбросить выбросы
# Создать фрейм данных
# houses = pd.DataFrame()
# houses['Цена'] = [534433, 392333, 293222, 4322032]
# houses['Ванные'] = [2, 3.5, 2, 116]
# houses['Кв_футы'] = [1500, 2500, 1500, 48000]
# Отфильтровать наблюдения
# houses[houses['Ванные'] < 20]
# 2. пометить их как выбросы и включить их в качестве признака
# Создать признак на основе булева условия
# houses["Выброс"] = np.where(houses["Ванные"] < 20, 0, 1)
# 3. преобразовать признак, чтобы ослабить эффект выброса
# Взять логарифм признака
# houses["Логарифм_кв_футов"] = [np.log(х) for х in houses["Кв_футы"]]

# Дан числовой признак, и требуется разбить его на дискретные корзины
from sklearn.preprocessing import Binarizer
# Создать признак
# age = np.array([[6],[12],[0],[36],[65]])
# Создать бинаризатор
# binarizer = Binarizer(18)
# Преобразовать признак
# binarizer.fit_transform(age)
# или разбить числовые признаки в соответствии с несколькими порогами
# Разнести признак по корзинам
# np.digitize(age, bins=[20,30,64])

# сгруппировать наблюдения так, чтобы похожие наблюдения находились в одной группе
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Создать матрицу симулированных признаков
# xc, _ = make_blobs(n_samples=50, n_features=2, centers=3, random_state=1)
# Создать фрейм данных
# dataframe = pd.DataFrame(xc, columns=["признак_1", "признак_2"])
# Создать кластеризатор по методу к средних
# clusterer = KMeans(3, random_state=0)
# Выполнить подгонку кластеризатора
# clusterer.fit(xc)
# Предсказать значения
# dataframe["группа"] = clusterer.predict(xc)

# удалить наблюдения, содержащие пропущенные значения
# Оставить только те наблюдения, которые не (помечены ~) пропущены
# xc[~np.isnan(xc).any(axis=1)]
# или
# Удалить наблюдения с отсутствующими значениями
# dataframe.dropna()

# имеются пропущенные значения, и требуется заполнить или предсказать их значения.
# Если объем данных небольшой, то предсказать пропущенные значения с помощью к ближайших соседей (KNN)
# Загрузить библиотеки
from fancyimpute import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
# Создать матрицу симулированных признаков
# xc, _ = make_blobs(n_samples = 1000, n_features = 2,random_state= 1)
# Стандартизировать признаки
# scaler = StandardScaler()
# standardized_features = scaler.fit_transform(xc)
# Заменить первое значение первого признака на пропущенное значение
# true_value = standardized_features[0,0]
# standardized_features[0,0] = np.nan
# Предсказать пропущенные значения в матрице признаков
# features_knn_imputed = KNN(k=5, verbose=0).complete(standardized_features)
# Сравнить истинные и импутированные значения
# print("Истинное значение:", true_value)
# print("Импутированное значение:", features_knn_imputed[0,0])