# метод главных компонент(principal component analysis, PCA)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Метод главных компонент — метод обучения без учителя(для понижения размерности данных)
# rng = np.random.RandomState(1)
# X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
# plt.scatter(X[:, 0], X[:, 1])
# plt.axis('equal')
# plt.show()
# В методе выполняется количественная оценка этой зависимости путем нахождения списка главных осей координат(principal axes)
# данных и их использования для описания набора данных
from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(X)
# алгоритм определяет величины(важные из них) — компоненты и объяснимая дисперсия
# print(pca.components_)
# def draw_vector(v0, v1, ax=None):
#     ax = ax or plt.gca()
#     arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0, shrinkB=0)
#     ax.annotate('', v1, v0, arrowprops=arrowprops)
# Рисуем данные
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# for length, vector in zip(pca.explained_variance_, pca.components_):
#     v = vector * 3 * np.sqrt(length)
#     draw_vector(pca.mean_, pca.mean_ + v)
# plt.axis('equal')
# plt.show()  # Визуализация главных осей данных. длина соответствует «важности» роли оси при описании распределения
# данных - это мера дисперсии данных при проекции на эту ось

# PCA как метод понижения размерности - обнуление одной или нескольких из наименьших главных компонент
# данные проецируются на пространство меньшей размерности с сохранением максимальной дисперсии данных
# pca = PCA(n_components=1)
# pca.fit(X)
# X_pca = pca.transform(X)
# print("original shape: ", X.shape)  # данные стали одномерными.
# print("transformed shape:", X_pca.shape)  # данные стали одномерными.
# X_new = pca.inverse_transform(X_pca)
# plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
# plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
# plt.axis('equal')
# plt.show()  # светлые точки — исходные данные, темные — спроецированная версия

# Использование метода PCA для визуализации: рукописные цифры
from sklearn.datasets import load_digits
# digits = load_digits()
# pca = PCA(2)  # Проекция из 64-мерного в двумерное пространство
# projected = pca.fit_transform(digits.data)
# график двух главных компонент каждой точки
# plt.scatter(projected[:, 0], projected[:, 1], c=digits.target,
#             edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('Spectral', 10))
# plt.xlabel('component 1')  # Компонента 1
# plt.ylabel('component 2')  # Компонента 2
# plt.colorbar()
# plt.show()  # эти точки — проекции каждой из точек данных вдоль направлений максимальной дисперсии

# Выбор количества компонент
# составная часть использования метода PCA — оценка количества компонент, необходимого для описания данных
# Определить это можно с помощью представления интегральной доли объяснимой дисперсии(explained variance ratio)
# pca = PCA().fit(digits.data)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()  # количественную оценку содержания общей, 64-мерной, дисперсии в первых N-компонентах
# первые 10 компонент содержат ~ 75% дисперсии, для описания ~ 100% доли дисперсии необходимо около 50 компонент

# Использование метода PCA для фильтрации шума
# шум должен довольно мало влиять на компоненты с дисперсией, значительно превышающей его уровень
# def plot_digits(data):
#     fig, axes = plt.subplots(4, 10, figsize=(10, 4), subplot_kw={'xticks':[], 'yticks': []},
#                              gridspec_kw=dict(hspace=0.1, wspace=0.1))
#     for i, ax in enumerate(axes.flat):
#         ax.imshow(data[i].reshape(8, 8), cmap='binary',
#                   interpolation='nearest', clim=(0, 16))
#     plt.show()
# plot_digits(digits.data)  # Цифры без шума
# случайный шум для создания зашумленного набора данных
# np.random.seed(42)
# noisy = np.random.normal(digits.data, 4)
# plot_digits(noisy)  # Цифры с добавленным Гауссовым случайным шумом
# обучение алгоритма PCA на зашумленных данных, указав, что проекция должна сохранять 50% дисперсии
# pca = PCA(0.50).fit(noisy)
# cv = pca.n_components_  # 12 - 50% дисперсии соответствует 12 главным компонентам.
# вычисление этих компонент и воспользуемся обратным преобразованием для восстановления отфильтрованных цифр
# components = pca.transform(noisy)
# filtered = pca.inverse_transform(components)
# plot_digits(filtered)  # Цифры с устраненным с помощью метода PCA шумом

# Пример: метод Eigenfaces
from sklearn.datasets import fetch_lfw_people
# faces = fetch_lfw_people(min_faces_per_person=60)
# какие главные оси координат охватывают этот набор данных
# RandomizedPCA — рандомизированный метод позволяет аппроксимировать первые N компонент быстрее,
# чем обычный оцениватель PCA(удобно для многомерных данных)
from sklearn.decomposition import PCA as RandomizedPCA
# pca = RandomizedPCA(150)
# pca.fit(faces.data)
# fig, axes = plt.subplots(3, 8, figsize=(9, 4), subplot_kw={'xticks': [], 'yticks': []},
#                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')
# plt.show()  # Визуализация собственных лиц, полученных из набора данных LFW
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()  # Интегральная объяснимая дисперсия для набора данных LFW
# сравнение входных изображений с восстановленными из 150 компонент(отвечают за более чем 90% дисперсии)
# Вычисляем компоненты и проекции лиц
# pca = RandomizedPCA(150).fit(faces.data)
# components = pca.transform(faces.data)
# projected = pca.inverse_transform(components)
# Рисуем результаты
# fig, ax = plt.subplots(2, 10, figsize=(10, 2.5),subplot_kw={'xticks': [], 'yticks': []},
#                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
# for i in range(10):
#     ax[0, i].imshow(faces.data[i].reshape(62, 47), cmap='binary_r')
#     ax[1, i].imshow(projected[i].reshape(62, 47), cmap='binary_r')
# Полноразмерные входные данные
# ax[0, 0].set_ylabel('full-dim\ninput')
# 150-мерная реконструкция
# ax[1, 0].set_ylabel('150-dim\nreconstruction')
# plt.show()  # 150-мерная реконструкция данных из LFW с помощью метода PCA
