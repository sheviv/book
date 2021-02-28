# Основы API статистического оценивания
# 1. Выбор класса модели с помощью импорта соответствующего класса оценивателя
# 2. Выбор гиперпараметров модели путем создания экземпляра этого класса с соответствующими значениями
# 3. Компоновка данных в матрицу признаков и целевой вектор
# 4. Обучение модели на своих данных вызова метода fit() экземпляра модели
# 5. Применение модели к новым данным:
#   - машинное обучение с учителем метки для неизвестных данных предсказывают с помощью метода predict()
#   - машинное обучение без учителя выполняется преобразование свойств данных или вывод значений методом transform()/predict()

# Пример обучения с учителем: простая линейная регрессия
import matplotlib.pyplot as plt
import numpy as np
# rng = np.random.RandomState(42)
# x = 10 * rng.rand(50,)
# y = 2 * x - 1 + rng.randn(50,)
# plt.scatter(x, y)
# plt.show()
# 1. Выбор класса модели
# from sklearn.linear_model import LinearRegression
# 2. Выбор гиперпараметров модели
# model = LinearRegression(fit_intercept=True)  # подбор точки пересечения с осью координат
# 3. Формирование из данных матриц признаков и целевого вектора
# X = x[:, np.newaxis]
# 4. Обучение модели на наших данных
# model.fit(X, y)
# 5. Предсказание меток для новых данных
# xfit = np.linspace(-1, 11)
# xfit преобразовать в матрицу признаков
# Xfit = xfit[:, np.newaxis]
# yfit = model.predict(Xfit)
# визуализируем результаты, нарисовав сначала график исходных данных, а затем обученную модель
# plt.scatter(x, y)
# plt.plot(xfit, yfit)
# plt.show()

# Пример обучения с учителем: классификация
# 1. разделить данные на обучающую последовательность (training set) и контрольную последовательность (testing set)
from sklearn.model_selection import train_test_split
import seaborn as sns
# iris = sns.load_dataset('iris')
# X_iris = iris.drop('species', axis=1)
# y_iris = iris['species']
# Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
# После упорядочения данных последуем нашему рецепту для предсказания меток
# 1. Выбираем класс модели
from sklearn.naive_bayes import GaussianNB
# 2. Создаем экземпляр модели
# model = GaussianNB()
# 3. Обучаем модель на данных
# model.fit(Xtrain, ytrain)
# 4. Предсказываем значения для новых данных
# y_model = model.predict(Xtest)
# accuracy_score для выяснения какая часть предсказанных меток соответствует истинному значению
from sklearn.metrics import accuracy_score
# print(accuracy_score(ytest, y_model))

# Пример обучения без учителя: понижение размерности набора данных
# 1. Выбираем класс модели
from sklearn.decomposition import PCA
# 2. Создаем экземпляр модели с гиперпараметрами
# model = PCA(n_components=2)
# 3. Обучаем модель на данных. Обратите внимание, что y мы не указываем!
# model.fit(X_iris)
# 4. Преобразуем данные в двумерные
# X_2D = model.transform(X_iris)
# iris['PCA1'] = X_2D[:, 0]
# iris['PCA2'] = X_2D[:, 1]
# sns.lmplot("PCA1", "PCA2", hue='species', data=iris, fit_reg=False)

# Обучение без учителя: кластеризация набора данных
# GMM - попытка моделирования данных в виде набора Гауссовых пятен
# 1. Выбираем класс модели
from sklearn.mixture import GaussianMixture as GMM
# 2. Создаем экземпляр модели с гиперпараметрами
# model = GMM(n_components=3, covariance_type='full')
# 3. Обучаем модель на данных. Обратите внимание, что y мы не указываем!
# model.fit(X_iris)
# 4. Определяем метки кластеров
# y_gmm = model.predict(X_iris)
# iris['cluster'] = y_gmm
# sns.lmplot(x="PCA1", y="PCA2", data=iris, hue='species', col='cluster', fit_reg=False)
# plt.show()

# Прикладная задача: анализ рукописных цифр
# from sklearn.datasets import load_digits
# digits = load_digits()
# cv = digits.images.shape  # 1797 выборок из сетки пикселов размером 8 × 8
# fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks':[], 'yticks':[]},gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
#     ax.text(0.05, 0.05, str(digits.target[i]),transform=ax.transAxes, color='green')
# «расплющим» массивы пикселов каждую цифру представлял массив пикселов длиной 64
# X = digits.data
# y = digits.target

# Обучение без учителя: понижение размерности
# данные станут двумерными
# from sklearn.manifold import Isomap
# iso = Isomap(n_components=2)
# iso.fit(digits.data)
# data_projected = iso.transform(digits.data)
# plt.scatter(data_projected[:, 0], data_projected[:, 1],c=digits.target, edgecolor='none',alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
# plt.colorbar(label='digit label', ticks=range(10))
# plt.clim(-0.5, 9.5)
# plt.show()