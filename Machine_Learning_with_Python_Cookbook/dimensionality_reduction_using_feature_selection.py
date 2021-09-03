# ch_10_Снижение размерности с помощью отбора признаков

# набор числовых признаков, и удалить те из них, которые имеют низкую дисперсию(скорее всего, содержат мало информации)
# Отобрать подмножество признаков с дисперсиями выше заданного порога
# Загрузить библиотеки
from sklearn import datasets
from sklearn.feature_selection import VarianceThreshold
# Импортировать немного данных для экспериментирования
# iris = datasets.load_iris()
# Создать признаки и цель
# features = iris.data
# target = iris.target
# Создать обработчик порога
# thresholder = VarianceThreshold(threshold=.5)
# Создать матрицу высокодисперсионных признаков
# features_high_variance = thresholder.fit_transform(features)
# Взглянуть на дисперсии
# thresholder.fit(features).variances_

# набор бинарных категориальных признаков, удалить те,которые имеют низкую дисперсию(содержат мало информации)
# Отобрать подмножество признаков с дисперсией бернуллиевых случайных величин выше заданного порога
# Загрузить библиотеку
from sklearn.feature_selection import VarianceThreshold
# Создать матрицу признаков, где:
# признак 0: 80% класс 0
# признак 1: 80% класс 1
# признак 2: 60% класс 0, 40% класс 1
# features = [[0, 1, 0],[0 , 1 , 1 ],[0, 1, 0],[0 , 1 , 1 ],[ 1 , 0 , 0 ]]
# Выполнить пороговую обработку по дисперсии
# thresholder = VarianceThreshold(threshold=(.75 * (1 - .75)))
# thresholder.fit_transform(features)

# в матрице признаков некоторые признаки сильно коррелированы
# Использовать корреляционную матрицу для выполнения проверки на сильно коррелированные признаки
# Если существуют сильно коррелированные признаки - рассмотреть возможность исключения одного из коррелированных признаков
# Загрузить библиотеки
import pandas as pd
import numpy as np
# Создать матрицу признаков с высококоррелированными признаками
# features = np.array([[1, 1, 1],[2, 2, 0],[3, 3, 1],[4, 4, 0],[5, 5, 1],[6, 6, 0],[7, 7, 1],[8, 7, 0],[9, 7, 1]])
# Конвертировать матрицу признаков во фрейм данных
# dataframe = pd.DataFrame(features)
# Создать корреляционную матрицу
# corr_matrix = dataframe.corr().abs()
# Выбрать верхний треугольник корреляционной матрицы
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
# Найти индекс столбцов признаков с корреляцией больше 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Исключить признаки
# dataframe.drop(dataframe.columns[to_drop], axis=1)
# сильно коррелированные признаки - удалить один из них из набора признаков

# удалить неинформативные признаки из категориального вектора целей
# Если признаки категориальные - вычислить статистический показатель хи-квадрат между каждым признаком и вектором целей
# Загрузить библиотеки
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
# Загрузить данные
# iris = load_iris()
# features = iris.data
# target = iris.target
# Конвертировать в категориальные данные путем преобразования данных в целые числа
# features = features.astype(int)
# Отобрать два признака с наивысшими значениями статистического показателя хи-квадрат
# chi2_selector = SelectKBest(chi2, k=2)
# features_kbest = chi2_selector.fit_transform(features, target)
# Показать результаты
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_kbest.shape[1])
# Если признаки являются количественными - вычислить статистический показатель
# F дисперсионного анализа(ANalysis O f VAriance, ANOVA) между признаком и вектором целей
# Отобрать два признака с наивысшими значениями статистического показателя F
# fvalue_selector = SelectKBest(f_classif, k=2)
# features_kbest = fvalue_selector.fit_transform(features, target)
# Показать результаты
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_kbest.shape[1])
# отбора верхнего п процента признаков
# Загрузить библиотеку
from sklearn.feature_selection import SelectPercentile
# Отобрать верхние 75% признаков с наивысшими значениями статистического показателя F
# fvalue_selector = SelectPercentile(f_classif, percentile=75)
# features_kbest = fvalue_selector.fit_transform(features, target)
# Показать результаты
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_kbest.shape[1])

# автоматически отобрать наилучшие признаки для использования в дальнейшем
# Для рекурсивного устранения признаков(recursive feature elimination,RFE) использовать класс RFECV(scikit-lean)
# с перекрестной проверкой(cross-validation, CV).
# Загрузить библиотеки
import warnings
from sklearn.datasets import make_regression
from sklearn.feature_selection import RFECV
from sklearn import datasets, linear_model
# Убрать раздражающее, но безвредное предупреждение
# warnings.filterwarnings(action="ignore", module="scipy",message="Ainternal gelsd")
# Сгенерировать матрицу признаков, вектор целей и истинные коэффициенты
# features, target = make_regression(n_samples = 10000,n_features = 100,n_informative = 2,random_state = 1)
# Создать объект линейной регрессии
# ols = linear_model.LinearRegression()
# Рекурсивно устранить признаки
# rfecv = RFECV(estimator=ols, step=1, scoring="neg_mean_squared_error")
# rfecv.fit(features, target)
# rfecv.transform(features)
# Количество наилучших признаков
# rfecv.n_features_
# Какие категории самые лучшие
# rfecv.support_
# Ранжировать признаки от самого лучшего (1) до самого плохого
# rfecv.ranking_
