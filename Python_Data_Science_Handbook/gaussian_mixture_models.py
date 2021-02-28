# смеси Гауссовых распределений
# смеси Гауссовых распределений, можно рассматривать в качестве развития идей метода k-средних

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Генерируем данные
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=400, centers=4,cluster_std=0.60, random_state=0)
X = X[:, ::-1]  # Транспонируем для удобства оси координат
# Выводим данные на график с полученными методом k-средних метками
from sklearn.cluster import KMeans
# kmeans = KMeans(4, random_state=0)
# labels = kmeans.fit(X).predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
# plt.show()  # Полученные методом k-средних метки для простых данных
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
#     labels = kmeans.fit_predict(X)
    # Выводим на рисунок входные данные
    # ax = ax or plt.gca()
    # ax.axis('equal')
    # ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    # Выводим на рисунок представление модели k-средних
    # centers = kmeans.cluster_centers_
    # radii = [cdist(X[labels == i], [center]).max() for i, center in enumerate(centers)]
    # for c, r in zip(centers, radii):
    #     ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5,zorder=1))
# kmeans = KMeans(n_clusters=4, random_state=0)
# plot_kmeans(kmeans, X)
# plt.show()  # Круглые кластеры, подразумеваемые моделью k-средних
# преобразовать те же данные, присвоенные метки окажутся перепутаны
# rng = np.random.RandomState(13)
# X_stretched = np.dot(X, rng.randn(2, 2))
# kmeans = KMeans(n_clusters=4, random_state=0)
# plot_kmeans(kmeans, X_stretched)
# plt.show()  # Неудовлетворительная работа метода k-средних в случае кластеров некруглой формы

# Обобщение EM-модели: смеси Гауссовых распределений(gaussian mixture model, GMM)
# поиск многомерных Гауссовых распределений вероятностей,
# моделирующих наилучшим возможным образом любой исходный набор данных.
# Смеси можно использовать для поиска кластеров аналогично методу k-средних
from sklearn.mixture import GaussianMixture as GMM
# gmm = GMM(n_components=4).fit(X)
# labels = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
# plt.show()  # Метки смеси Гауссовых распределений для данных
# можно также присваивать метки кластеров на вероятностной основе
# возвращает матрицу размера[n_samples, n_clusters],содержащую оценки вероятностей принадлежности точки к кластеру
# probs = gmm.predict_proba(X)

# функция для упрощения визуализации расположений и форм кластеров метода GMM
# путем рисования эллипсов на основе получаемой на выходе gmm информации
from matplotlib.patches import Ellipse
# def draw_ellipse(position, covariance, ax=None, **kwargs):
#     """Рисует эллипс с заданными расположением и ковариацией"""
#     ax = ax or plt.gca()
#     # Преобразуем ковариацию к главным осям координат
#     if covariance.shape == (2, 2):
#         U, s, Vt = np.linalg.svd(covariance)
#         angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
#         width, height = 2 * np.sqrt(s)
#     else:
#         angle = 0
#         width, height = 2 * np.sqrt(covariance)
#     # Рисуем эллипс
#     for nsig in range(1, 4):
#         ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, **kwargs))
# def plot_gmm(gmm, X, label=True, ax=None):
#     ax = ax or plt.gca()
#     labels = gmm.fit(X).predict(X)
#     if label:
#         ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
#     else:
#         ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
#         ax.axis('equal')
#     w_factor = 0.2 / gmm.weights_.max()
#     for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
#         draw_ellipse(pos, covar, alpha=w * w_factor)
# какие результаты выдает четырехкомпонентный метод GMM на данных
# gmm = GMM(n_components=4, random_state=42)
# plot_gmm(gmm, X)
# plt.show()  # Четырехкомпонентный метод GMM в случае круглых кластеров
# воспользоваться подходом GMM для «растянутого» набора данных.
# С учетом полной ковариации модель будет подходить даже для очень продолговатых, вытянутых в длину кластеров

# GMM как метод оценки плотности распределения
from sklearn.datasets import make_moons
# Xmoon, ymoon = make_moons(200, noise=.05, random_state=0)
# plt.scatter(Xmoon[:, 0], Xmoon[:, 1])
# plt.show()  # Использование метода GMM для кластеров с нелинейными границами
# попытаться использовать для этих данных двухкомпонентный GMM
# gmm2 = GMM(n_components=2, covariance_type='full', random_state=0)
# plot_gmm(gmm2, Xmoon)
# plt.show()  # Двухкомпонентная GMM-аппроксимация в случае нелинейных кластеров
# если взять намного больше компонент и проигнорировать метки кластеров,получится более подходящая для данных аппроксимация
# gmm16 = GMM(n_components=16, covariance_type='full', random_state=0)
# plot_gmm(gmm16, Xmoon, label=False)
# plt.show()  # Использование большого количества кластеров GMM для моделирования распределения точек

# Пример: использование метода GMM для генерации новых данных
from sklearn.datasets import load_digits
# digits = load_digits()
# def plot_digits(data):
#     fig, ax = plt.subplots(10, 10, figsize=(8, 8), subplot_kw=dict(xticks=[], yticks=[]))
#     fig.subplots_adjust(hspace=0.05, wspace=0.05)
#     for i, axi in enumerate(ax.flat):
#         im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
#         im.set_clim(0, 16)
# plot_digits(digits.data)
# plt.show()
# начнем с применения обратимого алгоритма для понижения размерности данных
from sklearn.decomposition import PCA
# pca = PCA(0.99, whiten=True)
# data = pca.fit_transform(digits.data)
# критерий AIC для определения необходимого количества компонент GMM
# n_components = np.arange(50, 210, 10)
# models = [GMM(n, covariance_type='full', random_state=0) for n in n_components]
# aics = [model.fit(data).aic(data) for model in models]
# plt.plot(n_components, aics)
# plt.show()
# gmm = GMM(110, covariance_type='full', random_state=0)
# gmm.fit(data)
# print(gmm.converged_)
# сгенерировать 100 новых точек в этом 41-мерном пространстве, используя GMM как порождающую модель
# data_new = gmm.sample(100, random_state=0)
# data_new = gmm.sample(100)
# обратным преобразованием объекта PCA для формирования новых цифр
# digits_new = pca.inverse_transform(data_new)
# plot_digits(digits_new)