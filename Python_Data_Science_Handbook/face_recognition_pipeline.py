# распознавания лиц

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Признаки в методе HOG(Гистограмма направленных градиентов) - процедура выделения признаков, для идентификации пешеходов.
# Метод HOG:
    # 1. предварительная нормализация изображений. получаются признаки, слабо зависящие от изменений освещенности.
    # 2. свертывания изображения с помощью двух фильтров, чувствительных к горизонтальным и вертикальным градиентам яркости.
    # Это улавливает информацию о границах, контурах и текстурах изображения.
    # 3. Разбивка изображения на ячейки заранее определенного размера и вычисление гистограммы направлений градиентов в каждой из ячеек.
    # 4. Нормализация гистограмм в каждой из ячеек(сравнением с несколькими близлежащими ячейками).
    # Это подавляет влияние освещенности на изображение.
    # 5. Формирование одномерного вектора признаков из информации по каждой ячейке.
# процедура выделения признаков на основе HOG
from skimage import data, color, feature
from skimage.feature import hog
import skimage.data
# image = color.rgb2gray(data.chelsea())
# hog_vec, hog_vis = feature.hog(image, visualise=True)
# hog_vec, hog_vis = hog(image, visualize=True)
# fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(xticks=[], yticks=[]))
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('input image')
# ax[1].imshow(hog_vis)
# ax[1].set_title('visualization of HOG features')
# plt.show()  # Визуализация HOG-признаков, вычисленных для изображения

# Метод HOG в действии: простой детектор лиц
# Алгоритм включает следующие шаги.
#     1. Получение миниатюр изображений, на которых представлены лица, для формирования набора «положительных» обучающих выборок.
#     2. Получение миниатюр изображений, на которых не представлены лица для формирования набора «отрицательных» обучающих выборок.
#     3. Выделение HOG-признаков из этих обучающих выборок.
#     4. Обучение линейного SVM-классификатора на этих выборках.
#     5. В случае «незнакомого» изображения перемещаем по изображению скользящее окно,
#     применяя нашу модель для определения того, содержится ли в этом окне лицо или нет.
#     6. Если обнаруженные лица частично пересекаются, объединяем их в одно окно.
# Получаем набор положительных обучающих выборок.
# from sklearn.datasets import fetch_lfw_people
# faces = fetch_lfw_people()
# positive_patches = faces.images
# Получаем набор отрицательных обучающих выборок. необходимо найти набор миниатюр такого же размера, на которых не изображены лица.
from skimage import data, transform
# imgs_to_use = ['camera', 'text', 'coins', 'moon', 'page', 'clock',
#                'immunohistochemistry', 'chelsea', 'coffee', 'hubble_deep_field']
# images = [color.rgb2gray(getattr(data, name)()) for name in imgs_to_use]
# from sklearn.feature_extraction.image import PatchExtractor
# def extract_patches(img, N, scale=1.0,
#     patch_size=positive_patches[0].shape):
#     extracted_patch_size = \
#     tuple((scale * np.array(patch_size)).astype(int))
#     extractor = PatchExtractor(patch_size=extracted_patch_size, max_patches=N, random_state=0)
#     patches = extractor.transform(img[np.newaxis])
#     if scale != 1:
#         patches = np.array([transform.resize(patch, patch_size) for patch in patches])
#     return patches
# negative_patches = np.vstack([extract_patches(im, 1000, scale) for im in images for scale in [0.5, 1.0, 2.0]])
# fig, ax = plt.subplots(6, 10)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(negative_patches[500 * i], cmap='gray')
#     axi.axis('off')
# plt.show()  # Отрицательные фрагменты изображений, не содержащие лиц
# Объединяем наборы и выделяем HOG-признаки. При наличии положительных и отрицательных выборок мы можем их объединить и вычислить HOG-признаки.
# from itertools import chain
# X_train = np.array([feature.hog(im) for im in chain(positive_patches, negative_patches)])
# y_train = np.zeros(X_train.shape[0])
# y_train[:positive_patches.shape[0]] = 1
# Обучние метода опорных векторов. воспользуемся простым Гауссовым наивным байесовским классификатором, чтобы было с чем сравнивать
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
# cv = cross_val_score(GaussianNB(), X_train, y_train)  # более чем 90%-ной точности.
# метод опорных векторов с поиском по сетке из нескольких вариантов параметра C
# from sklearn.svm import LinearSVC
# from sklearn.model_selection import GridSearchCV
# grid = GridSearchCV(LinearSVC(max_iter=1000), {'C': [1.0, 2.0, 4.0, 8.0]})
# grid.fit(X_train, y_train)
# Обучим оптимальный оцениватель на полном наборе данных
# model = grid.best_estimator_
# model.fit(X_train, y_train)
# Выполняем поиск лиц в новом изображении.
# test_image = skimage.data.astronaut()
# test_image = skimage.color.rgb2gray(test_image)
# test_image = skimage.transform.rescale(test_image, 0.5)
# test_image = test_image[:160, 40:180]
# plt.imshow(test_image, cmap='gray')
# plt.axis('off')
# plt.show()  # Изображение, в котором мы попытаемся найти лицо
# созда окно - будет перемещаться по фрагментам изображения с вычислением HOG-признаков для каждого фрагмента
# def sliding_window(img, patch_size=positive_patches[0].shape,
#     istep=2, jstep=2, scale=1.0):
#     Ni, Nj = (int(scale * s) for s in patch_size)
#     for i in range(0, img.shape[0] - Ni, istep):
#         for j in range(0, img.shape[1] - Ni, jstep):
#             patch = img[i:i + Ni, j:j + Nj]
#             if scale != 1:
#                 patch = transform.resize(patch, patch_size)
#             yield (i, j), patch
# indices, patches = zip(*sliding_window(test_image))
# patches_hog = np.array([feature.hog(patch) for patch in patches])
# взять фрагменты, для которых вычислены признаки HOG, и воспользуемся моделью, чтобы определить, содержат ли какие-то из них лица
# labels = model.predict(patches_hog)
# asd = labels.sum()  # 33 faces
# определить, где в контрольном изображении располагаются, нарисовав их границы в виде прямоугольников
# fig, ax = plt.subplots()
# ax.imshow(test_image, cmap='gray')
# ax.axis('off')
# Ni, Nj = positive_patches[0].shape
# indices = np.array(indices)
# for i, j in indices[labels == 1]:
#     ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red', alpha=0.3, lw=2, facecolor='none'))
# plt.show()  # Окна, в которых были обнаружены лица

