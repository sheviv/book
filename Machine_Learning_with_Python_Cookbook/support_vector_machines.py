# ch_17_Опорно-векторные машины

# Опорно-векторные машины классифицируют данные путем нахождения гиперплоскости,
# которая максимизирует допустимый промежуток между классами в тренировочных данных

# Тренировка линейного классификатора(натренировать модель, чтобы классифицировать наблюдения)
# опорно-векторный классификатор(support vector classifier, SVC), чтобы найти гиперплоскость(максимизирует
# промежутки(зазоры) между классами

from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
# Загрузить данные всего с двумя классами и двумя признаками
# iris = datasets.load_iris()
# features = iris.data[:100,:2]
# target = iris.target[:100]
# Стнадартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать опорно-векторный классификатор
# svc = LinearSVC(C=1.0)
# Натренировать модель
# model = svc.fit(features_standardized, target)
# Загрузить библиотеку
# from matplotlib import pyplot as pit
# Вывести точки данных на график и расцветить, используя их класс
# color = ["black" if c == 0 else "lightgrey" for c in target]
# pit.scatter(features_standardized[:, 0],
# features_standardized[:, 1], c=color)
# Создать гиперплоскость
# w = svc.coef_[0]
# a = -w[0] / w[1]
# xx = np.linspace(-2.5, 2.5)
# yy = a * xx - (svc.intercept_[0]) / w[1]
# Начертить гиперплоскость
# pit.plot(xx, yy)
# pit.axis("off")
# pit.show()

# Обработка линейно неразделимых классов с помощью ядер(натренировать опорно-векторный классификатор, но классы линейно неразделимы)
# Натренировать расширение опорно-векторной машины с использованием ядерных функций для создания нелинейных границ решений
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
# Задать начальное значение рандомизации
# np.random.seed(0)
# Сгенерировать два признака
# features = np.random.randn(200, 2)
# Применить вентиль XOR, чтобы сгенерировать линейно разделимые классы
# target_xor = np.logical_xor(features[:, 0] > 0, features[:, 1] > 0)
# target = np.where(target_xor, 0, 1)
# Создать опорно-векторную машину с радиально-базисным функциональным ядром (RBF-ядром)
# svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)
# Натренировать классификатор
# model = svc.fit(features, target)
# Вывести на график наблюдения и гиперплоскость границы решения
# from matplotlib.colors import ListedColormap
# import matplotlib.pyplot as plt
# def plot_decision_regions(X, y, classifier):
#     cmap = ListedColormap(("red", "blue"))
#     xx1, xx2 = np.meshgrid(np.arange(-3, 3, 0.02), np.arange(-3, 3, 0.02))
#     Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
#     Z = Z.reshape(xx1.shape)
#     plt.contourf(xx1, xx2, Z, alpha=0.1, cmap=cmap)
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx),marker="+", label=cl)
# Создать опорно-векторный классификатор с линейным ядром
# svc_linear = SVC(kernel="linear", random_state=0, C=1)
# # Натренировать модель
# svc_linear.fit(features, target)
# SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3,
#     gamma='auto', kernel='linear',max_iter=-1, probability=False, random_state=0, shrinking=True,
#     tol=0.001, verbose=False)
# # Вывести на график наблюдения и гиперплоскость
# plot_decision_regions(features, target, classifier=svc_linear)
# plt.axis("off")
# plt.show()
# или
# Создать опорно-векторную машину с радиально-базисным функциональным ядром (RBF-ядром)
# svc = SVC(kernel="rbf", random_state=0, gamma=1, C=1)
# Натренировать классификатор
# model = svc.fit(features, target)
# Вывести на график наблюдения и гиперплоскость
# plot_decision_regions(features, target, classifier=svc)
# plt.axis("off")
# plt.show()

# Создание предсказанных вероятностей(узнать предсказанные вероятности класса для наблюдения)
# установить probabiiity=True,натренировать модель,затем для просмотра откалиброванных вероятностей применить predictjprobe
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект опорно-векторного классификатора
# svc = SVC(kernel="linear", probability=True, random_state=0)
# Натренировать классификатор
# model = svc.fit(features_standardized, target)
# Создать наблюдение
# new_observation = [[.4, .4, .4, .4]]
# Взглянуть на предсказанные вероятности
# model.predict_proba(new_observation)

# Идентификация опорных векторов(определить наблюдения являются опорными векторами гиперплоскости решения)
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
# Загрузить данные только с двумя классами
# iris = datasets.load_iris()
# features = iris.data[:100,:]
# target = iris.target[:100]
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект опорно-векторного классификатора
# svc = SVC(kernel="linear", random_state=0)
# Натренировать классификатор
# model = svc.fit(features_standardized, target)
# Взглянуть на опорные векторы
# model.support_vectors_

# Обработка несбалансированных классов(натренировать опорно-векторный классификатор в присутствии несбалансированных классов)
# увеличить штраф за ошибочное классифицирование миноритарных классов, чтобы они не были "подавлены" мажоритарным классом.
# С — это гиперпараметр, определяющий штраф за ошибочное классифицирование наблюдения.
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
# Загрузить данные только с двумя классами
# iris = datasets.load_iris()
# features = iris.data[:100,:]
# target = iris.target[:100]
# Сделать класс сильно несбалансированным, удалив первые 40 наблюдений
# features = features[40:,:]
# target = target[40:]
# Создать вектор целей, указав класс 0 либо 1
# target = np.where((target == 0), 0, 1)
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать опорно-векторный классификатор
# svc = SVC(kernel="linear", class_weight="balanced", C=1.0, random_state=0)
# Натренировать классификатор
# model = svc.fit(features_standardized, target)