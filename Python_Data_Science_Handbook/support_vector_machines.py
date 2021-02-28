# метод опорных векторов

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# Воспользуемся настройками по умолчанию библиотеки Seaborn
import seaborn as sns
sns.set()

# Вместо моделирования каждого из классов найдем прямую или кривую(в двумерном пространстве)
# или многообразие(в многомерном пространстве), отделяющее классы друг от друга.
from sklearn.datasets import make_blobs
# X, y = make_blobs(n_samples=50, centers=2,random_state=0, cluster_std=0.60)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.show()
# существует более одной идеально разделяющей два класса прямой
# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2,markersize=10)
# for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
#     plt.plot(xfit, m * xfit + b, '-k')
# plt.xlim(-1, 3.5)
# plt.show()  # показаны три очень разных разделителя,в зависимости от того, какой из них выберать,
# новой точке данных будут присвоены различные метки

# Метод опорных векторов: максимизируем отступ
# вместо того чтобы рисовать между классами прямую нулевой ширины, можно нарисовать около каждой - отступ(margin)
# некоторой ширины, простирающийся до ближайшей точки
# xfit = np.linspace(-1, 3.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
#     yfit = m * xfit + b
#     plt.plot(xfit, yfit, '-k')
#     plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', color='#AAAAAA', alpha=0.4)
# plt.xlim(-1, 3.5)
# plt.show()
# Метод опорных векторов — пример оценивателя с максимальным отступом(maximum margin estimator) -
# в качестве оптимальной модели выбирается линия, максимизирующая этот отступ.

# Аппроксимация методом опорных векторов
# линейное ядро и зададим очень большое значение параметра C
from sklearn.svm import SVC # "Классификатор на основе метода опорных векторов"
# model = SVC(kernel='linear', C=1E10)
# model.fit(X, y)
# функция для построения графика границ решений метода SVM
# def plot_svc_decision_function(model, ax=None, plot_support=True):
#     """Строим график решающей функции для двумерной SVC"""
#     if ax is None:
#         ax = plt.gca()
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     # Создаем координатную сетку для оценки модели
#     x = np.linspace(xlim[0], xlim[1], 30)
#     y = np.linspace(ylim[0], ylim[1], 30)
#     Y, X = np.meshgrid(y, x)
#     xy = np.vstack([X.ravel(), Y.ravel()]).T
#     P = model.decision_function(xy).reshape(X.shape)
#     # Рисуем границы принятия решений и отступы
#     ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
#     # Рисуем опорные векторы
#     if plot_support:
#         ax.scatter(model.support_vectors_[:, 0],
#                    model.support_vectors_[:, 1],
#                    s=300, linewidth=1, facecolors='none')
#     ax.set_xlim(xlim)
#     ax.set_ylim(ylim)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(model)
# plt.show()
# касающиеся прямой точки — ключевые элементы аппроксимации, они известны под названием опорных векторов

# Выходим за границы линейности: SVM-ядро
from sklearn.datasets import make_circles
# X, y = make_circles(100, factor=.1, noise=.1)
# clf = SVC(kernel='linear').fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf, plot_support=False)
# plt.show()
# r = np.exp(-(X ** 2).sum(1))
# clf = SVC(kernel='rbf', C=1E6)
# clf.fit(X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plot_svc_decision_function(clf)
# plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
# s=300, lw=1, facecolors='none')
# plt.show()

# Настройка SVM: размытие отступов(что, если данные в некоторой степени перекрываются?)
# X, y = make_blobs(n_samples=100, centers=2,random_state=0, cluster_std=1.2)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
# plt.show()
# параметр разрешает некоторым точкам «заходить» на отступ в тех случаях, когда это приводит к лучшей аппроксимации
# X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# for axi, C in zip(ax, [10.0, 0.1]):
#     model = SVC(kernel='linear', C=C).fit(X, y)
#     axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
#     plot_svc_decision_function(model, axi)
#     axi.scatter(model.support_vectors_[:, 0],
#     model.support_vectors_[:, 1],s=300, lw=1, facecolors='none')
#     axi.set_title('C = {0:.1f}'.format(C), size=14)
    # plt.show()

# Пример: распознавание лиц
# from sklearn.datasets import fetch_lfw_people
# faces = fetch_lfw_people(min_faces_per_person=60)
# print(faces.target_names)
# print(faces.images.shape)
# fig, ax = plt.subplots(3, 5)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(faces.images[i], cmap='bone')
#     axi.set(xticks=[], yticks=[],xlabel=faces.target_names[faces.target[i]])
# Упростим эту задачу, объединив препроцессор и классификатор в единый конвейер:
# from sklearn.svm import SVC
# from sklearn.decomposition import PCA as RandomizedPCA
# from sklearn.pipeline import make_pipeline
# pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
# svc = SVC(kernel='rbf', class_weight='balanced')
# model = make_pipeline(pca, svc)
# разобьем данные на обучающую и контрольную последовательности
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target,random_state=42)
# - воспользуемся поиском по сетке с перекрестной проверкой для анализа сочетаний параметров
# - подберем значения параметров C(размытие отступов) и gamma(размер ядра радиальной базисной функции)
# - определим оптимальную модель
# from sklearn.model_selection import GridSearchCV
# param_grid = {'svc__C': [1, 5, 10, 50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
# grid = GridSearchCV(model, param_grid)
# model = grid.best_estimator_
# yfit = model.predict(Xtest)
# plt.show()