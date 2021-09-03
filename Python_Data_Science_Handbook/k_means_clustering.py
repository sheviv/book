# кластеризация методом k-средних
# кластеризация - оптимальное разбиение или дискретную маркировку групп точек

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()


# k-средних - поиск заранее заданного количества кластеров в не маркированном многомерном наборе данных
#     - «Центр кластера» — арифметическое среднее всех точек, относящихся к этому кластеру
#     - Каждая точка ближе к центру своего кластера, чем к центрам других кластеров.
# Во-первых, сгенерируем двумерный набор данных, содержащий четыре отдельных «пятна»
from sklearn.datasets import make_blobs
# X, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)
# plt.scatter(X[:, 0], X[:, 1], s=50)
from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=4)
# kmeans.fit(X)
# y_kmeans = kmeans.predict(X)
# plt.show()  # Данные для демонстрации кластеризации
# Нарисуем найденные оценивателем метода k-средних центры кластеров
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
# centers = kmeans.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.show()  # Центры кластеров метода k-средних с окрашенными в разные цвета кластерами

# Алгоритм k-средних: максимизация математического ожидания
# Максимизация математического ожидания(EM):
#     - Выдвигаем гипотезу о центрах кластеров.
#     - Повторяем до достижения сходимости:
#         - E-шаг(шаг ожидания): приписываем точки к ближайшим центрам кластеров
#         - M-шаг(шаг максимизации): задаем новые центры кластеров в соответствии со средними значениями

# Пример: Алгоритм k-средних
from sklearn.metrics import pairwise_distances_argmin
# def find_clusters(X, n_clusters, rseed=2):
#     # 1. Выбираем кластеры случайным образом
#     rng = np.random.RandomState(rseed)
#     i = rng.permutation(X.shape[0])[:n_clusters]
#     centers = X[i]
#     while True:
#         # 2a. Присваиваем метки в соответствии с ближайшим центром
#         labels = pairwise_distances_argmin(X, centers)
#         # 2b. Находим новые центры, исходя из средних значений точек
#         new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
#         # 2c. Проверяем сходимость
#         if np.all(centers == new_centers):
#             break
#         centers = new_centers
#
#     return centers, labels
# centers, labels = find_clusters(X, 4)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.show()  # Данные, маркированные с помощью метода k-средних
# если воспользоваться в процедуре другим начальным значением для г.с.п.ч, начальные гипотезы приведут к плохим результатам
# centers, labels = find_clusters(X, 4, rseed=0)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.show()  # Пример плохой сходимости в методе k-средних
# Количество кластеров следует выбирать заранее.
# labels = KMeans(6, random_state=0).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.show()  # Пример неудачного выбора количества кластеров
# Применение метода k-средних ограничивается случаем линейных границ кластеров.
from sklearn.datasets import make_moons
# X, y = make_moons(200, noise=.05, random_state=0)
# labels = KMeans(2, random_state=0).fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# plt.show()  # Неудача метода k-средних в случае нелинейных границ
# граф ближайших соседей для вычисления представления данных более высокой размерности,
# после чего задает соответствие меток с помощью алгоритма k-средних
# from sklearn.cluster import SpectralClustering
# model = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',assign_labels='kmeans')
# labels = model.fit_predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels,s=50, cmap='viridis')
# plt.show()  # Нелинейные границы, обнаруженные с помощью SpectralClustering

# Пример 1: применение метода k-средних для рукописных цифр
from sklearn.datasets import load_digits
# digits = load_digits()
# # Кластеризация выполняется так же
# kmeans = KMeans(n_clusters=10, random_state=0)
# clusters = kmeans.fit_predict(digits.data)
# asd = kmeans.cluster_centers_.shape  # десять кластеров в 64-мерном пространстве.
# fig, ax = plt.subplots(2, 5, figsize=(8, 3))
# centers = kmeans.cluster_centers_.reshape(10, 8, 8)
# for axi, center in zip(ax.flat, centers):
#     axi.set(xticks=[], yticks=[])
#     axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)
# plt.show()  # Центры кластеров - список цифр
# k-средних не знает о сущности кластеров, метки 0–9 могут оказаться перепутаны местами
# зададим соответствие всех полученных меток кластеров имеющимся в них фактическим меткам
from scipy.stats import mode
# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(digits.target[mask])[0]
# насколько точно кластеризация без учителя определила подобие цифр в данных
from sklearn.metrics import accuracy_score
# accuracy_score(digits.target, labels)  # 80%
# матрица различий
from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(digits.target, labels)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=digits.target_names, yticklabels=digits.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()  # Матрица различий для классификатора методом k-средних
# t-SNE — нелинейный алгоритм вложения, хорошо умеющий сохранять точки внутри кластеров.
from sklearn.manifold import TSNE
# Проекция данных: выполнение этого шага займет несколько секунд
# tsne = TSNE(n_components=2, init='pca', random_state=0)
# digits_proj = tsne.fit_transform(digits.data)
# # Расчет кластеров
# kmeans = KMeans(n_clusters=10, random_state=0)
# clusters = kmeans.fit_predict(digits_proj)
# # Перестановка меток местами
# labels = np.zeros_like(clusters)
# for i in range(10):
#     mask = (clusters == i)
#     labels[mask] = mode(digits.target[mask])[0]
# # Оценка точности
# accuracy_score(digits.target, labels)  # 94 %

# Пример 2: использование метода k-средних для сжатия цветов
# Обратите внимание: для работы этого кода
# должен быть установлен пакет pillow
from sklearn.datasets import load_sample_image
# china = load_sample_image("china.jpg")
# ax = plt.axes(xticks=[], yticks=[])
# ax.imshow(china)
# plt.show()  # Исходное изображение
# Изменим форму данных на [n_samples × n_features] и масштабируем шкалу цветов, чтобы они располагались между 0 и 1
# data = china / 255.0 # используем шкалу 0...1
# data = data.reshape(427 * 640, 3)
# Визуализируем эти пикселы в данном цветовом пространстве, используя подмножество из 10 000 пикселов для быстроты работы
# def plot_pixels(data, title, colors=None, N=10000):
#     if colors is None:
#         colors = data
#     # Выбираем случайное подмножество
#     rng = np.random.RandomState(0)
#     i = rng.permutation(data.shape[0])[:N]
#     colors = colors[i]
#     R, G, B = data[i].T
#     fig, ax = plt.subplots(1, 2, figsize=(16, 6))
#     ax[0].scatter(R, G, color=colors, marker='.')
#     ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))
#     ax[1].scatter(R, B, color=colors, marker='.')
#     ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))
#     fig.suptitle(title, size=20)
# plot_pixels(data, title='Input color space: 16 million possible colors') # цветовое пространство: 16мл.возможных цветов
# plt.show()  # Распределение пикселов в цветовом пространстве RGB
# уменьшим кол-во цветов с 16мл. до 16 путем кластеризации методом k-средних на пространстве пикселов
# воспользуемся мини-пакетным методом k-средних(вычисляет результат быстрее, чем стандартный метод k-средних)
from sklearn.cluster import MiniBatchKMeans
# kmeans = MiniBatchKMeans(16)
# kmeans.fit(data)
# new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
# plot_pixels(data, colors=new_colors, title="Reduced color space: 16 colors")  # Редуцированное цвет. пространство: 16цвет
# plt.show()  # Шестнадцать кластеров в цветовом пространстве RGB
# эффект от перекрашивания
# china_recolored = new_colors.reshape(china.shape)
# fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw=dict(xticks=[], yticks=[]))
# fig.subplots_adjust(wspace=0.05)
# ax[0].imshow(china)
# ax[0].set_title('Original Image', size=16)  # Первоначальное изображение
# ax[1].imshow(china_recolored)
# ax[1].set_title('16-color Image', size=16)  # 16-цветное изображение
# plt.show()  #  Полноцветное изображение (слева) по сравнению с 16-цветным (справа)