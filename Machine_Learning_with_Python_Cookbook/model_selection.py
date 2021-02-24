# ch_12_Выбор модели

# отобрать наилучшую модель, выполнив поиск по диапазону гиперпараметров
# Загрузить библиотеки
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект логистической регрессии
# logistic = linear_model.LogisticRegression()
# Создать диапазон вариантов значений штрафного гиперпараметра
# penalty = ['l1', 'l2']
# Создать диапазон вариантов значений регуляризационного гиперпараметра
# C = np.logspace(0, 4, 10)
# Создать словарь вариантов гиперпараметров
# hyperparameters = dict(С=C, penalty=penalty)
# Создать объект решеточного поиска| verbose задает объем сообщений, выводимых во время поиска(0 - отсутствие сообщений,
# 1 до 3 — вывод сообщений с большей детализацией
# gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
# Выполнить подгонку объекта решеточного поиска
# best_model = gridsearch.fit(features, target)
# Взглянуть на наилучшие гиперпараметры
# print('Лучший штраф:', best_model.best_estimator_.get_params()['penalty'])
# print('Лучший C:', best_model.best_estimator_.get_params()['С'])
# Предсказать вектор целей
# best_model.predict(features)

# Отбор наилучших моделей с помощью рандомизированного поиска(вычислительно более дешевый метод)
# случайным образом отбирать без возврата образцы значений гиперпараметров из этого распределения
from scipy.stats import uniform
from sklearn import linear_model, datasets
from sklearn.model_selection import RandomizedSearchCV
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект логистической регрессии
# logistic = linear_model.LogisticRegression()
# Создать диапазон вариантов значений штрафного гиперпараметра
# penalty = ['l1', 'l2']
# Создать диапазон вариантов значений регуляризационного гиперпараметра
# C = uniform(loc=0, scale=4)
# Создать словарь вариантов гиперпараметров
# hyperparameters = dict(C=C, penalty=penalty)
# Создать объект рандомизированного поиска
# randomizedsearch = RandomizedSearchCV(logistic, hyperparameters, random_state=1, n_iter=100,cv=5, verbose=0, n_jobs=-1)
# Выполнить подгонку объекта рандомизированного поиска
# best_model = randomizedsearch.fit(features, target)
# Определить равномерное распределение между 0 и 4 отобрать 10 значений
# uniform(loc=0, scale=4).rvs(10)
# Взглянуть на лучшие гиперпараметры
# print('Лучший штраф:', best_model.best_estimator_.get_params()['penalty'])
# print('Лучший С:', best_model.best_estimator_.get_params()['C'])

# Отбор наилучших моделей из нескольких обучающихся алгоритмов
# (по диапазону обучающихся алгоритмов и их соответствующих гиперпараметров)
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
# Задать начальное число для генератора псевдослучайных чисел
# np.random.seed(0)
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать конвейер
# pipe = Pipeline([("classifier", RandomForestClassifier())])
# Создать словарь вариантов обучающихся алгоритмов и их гиперпараметров
# search_space = [{"classifier": [LogisticRegression()],"classifier__penalty": ['l1', 'l2'],
#                  "classifier__C": np.logspace(0, 4, 10)},{"classifier": [RandomForestClassifier()],
#                 "classifier__n_estimators": [10, 100, 1000], "classifier max features": [1, 2, 3]}],
# Создать объект решеточного поиска
# gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0)
# Выполнить подгонку объекта решеточного поиска
# best_model = gridsearch.fit(features, target)
# Взглянуть на лучшую модель
# best_model.best_estimator_.get_params()["classifier"]
# Предсказать вектор целей
# best_model.predict(features)

# отбор наилучших моделей во время предобработки/включить шаг предобработки во время отбора модели
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Задать начальное число для генератора псевдослучайных чисел
# np.random.seed(0)
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект предобработки, который включает признаки стандартного шкалировщика StandardScaler и объект РСА
# preprocess = FeatureUnion([("std", StandardScaler()), ("pea", PCA())])
# Создать конвейер
# pipe = Pipeline([("preprocess", preprocess), ("classifier", LogisticRegression())])
# Создать пространство вариантов значений
# search_space = [{"preprocess__pea n components": [1, 2, 3], "classifier__penalty": ["11", "12"],
#                  "classifier__C": np.logspace(0, 4, 10)}]
# Создать объект решеточного поиска
# elf = GridSearchCV(pipe, search_space, cv=5, verbose=0, n_jobs=-1)
# Выполнить подгонку объекта решеточного поиска
# best_model = elf.fit(features, target)
# Взглянуть на самую лучшую модель
# best_model.best_estimator_.get_params()['preprocess__рса__n_components']

# Ускорение отбора модели с помощью распараллеливания
# Использовать на компьютере все ядра центрального процессора, установив n_jobs=-1
# n_jobs задает количество моделей для параллельной тренировки
import numpy as пр
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект логистической регрессии
# logistic = linear_model.LogisticRegression()
# Создать диапазон вариантов значений штрафного гиперпараметра
# penalty = ["l1", "l2"]
# Создать диапазон вариантов значений регуляризационного гиперпараметра
# C = np.logspace(0, 4, 1000)
# Создать словарь вариантов гиперпараметров
# hyperparameters = dict(C=C, penalty=penalty)
# Создать объект решеточного поиска
# gridsearch = GridSearchCV(logistic, hyperparameters, cv=5,n_jobs=-1, verbose=1)
# Выполнить подгонку объекта решеточного поиска
# best_model = gridsearch.fit(features, target)

# Ускорение отбора модели с помощью алгоритмически специализированных методов
# Если используется отборное число обучающихся алгоритмов, применить модельно
# специфическую перекрестно-проверочную настройку гиперпараметров
from sklearn import linear_model, datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект перекрестно-проверяемой логистической регрессии
# logit = linear_model.LogisticRegressionCV(Cs=100)
# Натренировать модель
# logit.fit(features, target)

# оценить результативность модели, найденной с помощью отбора модели
# Использовать вложенную перекрестную проверку, чтобы избежать смещенной оценки
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV, cross_val_score
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект логистической регрессии
# logistic = linear_model.LogisticRegression()
# Создать диапазон из 20 вариантов значений для С
# C = np.logspace(0, 4, 20)
# Создать словарь вариантов гиперпараметров
# hyperparameters = dict(C=C)
# Создать объект решеточного поиска
# gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, n_jobs=-1, verbose=0)
# Выполнить вложенную перекрестную проверку и выдать среднюю оценку
# cross_val_score(gridsearch, features, target).mean()