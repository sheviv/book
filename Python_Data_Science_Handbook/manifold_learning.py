# обучение на базе многообразий
# класс оценивателей без учителя, нацеленных на описание наборов данных как низкоразмерных многообразий,
# вложенных в пространство большей размерности
# пример: Алгоритмы обучения на базе многообразий нацелены на изучение базовой двумерной природы листа бумаги,
# даже если он был смят ради размещения в трехмерном пространстве

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Обучение на базе многообразий: HELLO
# пример: генерации двумерных данных, подходящих для описания многообразия
# def make_hello(N=1000, rseed=42):
#     # Создаем рисунок с текстом "HELLO"; сохраняем его в формате PNG
#     fig, ax = plt.subplots(figsize=(4, 1))
#     fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     ax.axis('off')
#     ax.text(0.5, 0.4, 'HELLO', va='center', ha='center', weight='bold',size=85)
#     fig.savefig('hello.png')
#     plt.close(fig)
#     # plt.show()
#     # Открываем этот PNG-файл и берем из него случайные точки
#     from matplotlib.image import imread
#     data = imread('hello.png')[::-1, :, 0].T
#     rng = np.random.RandomState(rseed)
#     X = rng.rand(4 * N, 2)
#     i, j = (X * data.shape).astype(int).T
#     mask = (data[i, j] < 1)
#     X = X[mask]
#     X[:, 0] *= (data.shape[0] / data.shape[1])
#     X = X[:N]
#     return X[np.argsort(X[:, 0])]
# # визуализируем полученные данные
# X = make_hello(1000)
# colorize = dict(c=X[:, 0], cmap=plt.cm.get_cmap('rainbow', 5))
# plt.scatter(X[:, 0], X[:, 1], **colorize)
# plt.axis('equal')
# plt.show()  # Данные для обучения на базе многообразий

# Многомерное масштабирование (MDS)
# def rotate(X, angle):
#     theta = np.deg2rad(angle)
#     R = [[np.cos(theta), np.sin(theta)],
#     [-np.sin(theta), np.cos(theta)]]
#     return np.dot(X, R)
# X2 = rotate(X, 20) + 5
# plt.scatter(X2[:, 0], X2[:, 1], **colorize)
# plt.axis('equal')
# plt.show()  # Набор данных после вращения

# MDS как обучение на базе многообразий
# спроецировать их в трехмерное пространство
# def random_projection(X, dimension=3, rseed=42):
#     assert dimension >= X.shape[1]
#     rng = np.random.RandomState(rseed)
#     C = rng.randn(dimension, dimension)
#     e, V = np.linalg.eigh(np.dot(C, C.T))
#     return np.dot(X, V[:X.shape[1]])
# X3 = random_projection(X, 3)
# Визуализируем точки
from mpl_toolkits import mplot3d
# ax = plt.axes(projection='3d')
# ax.scatter3D(X3[:, 0], X3[:, 1], X3[:, 2], **colorize)
# ax.view_init(azim=70, elev=50)
# plt.show()  # Данные, линейно вложенные в трехмерное пространство
# and
# передать эти данные оценивателю MDS для вычисления матрицы расстояний и
# последующего определения оптимального двумерного вложения
from sklearn.manifold import MDS
# model = MDS(n_components=2, random_state=1)
# out3 = model.fit_transform(X3)
# plt.scatter(out3[:, 0], out3[:, 1], **colorize)
# plt.axis('equal')
# plt.show()  # MDS-вложение трехмерных данных позволяет восстановить исходные данные с точностью до вращения и отражения
# при заданных многомерных вложенных данных MDS находит низкоразмерное их представление,
# сохраняющее определенные зависимости внутри данных

# Нелинейные вложения: там, где MDS не работает
# в случае нелинейного вложения(при выходе за пределы этого простого набора операций)MDS терпит неудачу
# def make_hello_s_curve(X):
#     t = (X[:, 0] - 2) * 0.75 * np.pi
#     x = np.sin(t)
#     y = X[:, 1]
#     z = np.sign(t) * (np.cos(t) - 1)
#     return np.vstack((x, y, z)).T
# XS = make_hello_s_curve(X)
from mpl_toolkits import mplot3d
# ax = plt.axes(projection='3d')
# ax.scatter3D(XS[:, 0], XS[:, 1], XS[:, 2], **colorize)
# plt.show()  # Нелинейно вложенные в трехмерное пространство данные. свернуты в форму буквы S
from sklearn.manifold import MDS
# model = MDS(n_components=2, random_state=2)
# outS = model.fit_transform(XS)
# plt.scatter(outS[:, 0], outS[:, 1], **colorize)
# plt.axis('equal')
# plt.show()  # Использование алгоритма MDS для нелинейных данных: попытка восстановления исходной структуры - неудачна
# двумерное линейное вложение не сможет развернуть обратно S-образную кривую, а отбросит вместо этого исходную ось координат Y

# Нелинейные многообразия: локально линейное вложение
# модифицированный алгоритм LLE
from sklearn.manifold import LocallyLinearEmbedding
# model = LocallyLinearEmbedding(n_neighbors=100, n_components=2,method='modified',eigen_solver='dense')
# out = model.fit_transform(XS)
# fig, ax = plt.subplots()
# ax.scatter(out[:, 0], out[:, 1], **colorize)
# ax.set_ylim(0.15, -0.15)
# plt.show()  # Локально линейное вложение может восстанавливать изначальные данные из нелинейно вложенных входных данных

# Пример: использование Isomap для распознавания лиц
from sklearn.datasets import fetch_lfw_people
# faces = fetch_lfw_people(min_faces_per_person=30)
# asd = faces.data.shape  # (2370, 2914) - 2370 изображений по 2914 пикселов(точками данных в 2914-мерном пространстве)
# fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='gray')
# plt.show()  # Примеры исходных лиц
# сколько линейных признаков необходимо для описания этих данных
from sklearn.decomposition import PCA as RandomizedPCA
# model = RandomizedPCA(100).fit(faces.data)
# plt.plot(np.cumsum(model.explained_variance_ratio_))
# plt.xlabel('n components')  # Количество компонент
# plt.ylabel('cumulative variance')  # Интегральная дисперсия
# plt.show()  # Интегральная дисперсия, полученная из проекции методом PCA
# Рассчитать вложение Isomap
from sklearn.manifold import Isomap
# model = Isomap(n_components=2)
# proj = model.fit_transform(faces.data)
# ads = proj.shape  # двумерная проекция всех исходных изображений
# опишем функцию, выводящую миниатюры изображений в местах проекций
from matplotlib import offsetbox
# def plot_components(data, model, images=None, ax=None,thumb_frac=0.05, cmap='gray'):
#     ax = ax or plt.gca()
#     proj = model.fit_transform(data)
#     ax.plot(proj[:, 0], proj[:, 1], '.k')
#     if images is not None:
#         min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
#         shown_images = np.array([2 * proj.max(0)])
#         for i in range(data.shape[0]):
#             dist = np.sum((proj[i] - shown_images) ** 2, 1)
#             if np.min(dist) < min_dist_2:
#                 # Не отображаем слишком близко расположенные точки
#                 continue
#             shown_images = np.vstack([shown_images, proj[i]])
#             imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), proj[i])
#             ax.add_artist(imagebox)
# fig, ax = plt.subplots(figsize=(10, 10))
# plot_components(faces.data, model=Isomap(n_components=2), images=faces.images[:, ::2, ::2])
# plt.show()  # Вложение с помощью Isomap данных о лицах

