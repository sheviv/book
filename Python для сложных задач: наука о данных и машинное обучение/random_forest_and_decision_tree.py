# деревья решений и случайные леса

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# деревья принятия решений
# Создание дерева принятия решений
from sklearn.datasets import make_blobs
# X, y = make_blobs(n_samples=300, centers=4,random_state=0, cluster_std=1.0)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')
# plt.show()  # дерево принятия решений для этих данных будет многократно разделять данные по осям,
# в соответствии с определенным количественным критерием

# Процесс обучения дерева принятия решений
from sklearn.tree import DecisionTreeClassifier
# tree = DecisionTreeClassifier().fit(X, y)
# визуализация вывода классификатора
# def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
#     ax = ax or plt.gca()
#     # Рисуем обучающие точки
#     ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,clim=(y.min(), y.max()), zorder=3)
#     ax.axis('tight')
#     ax.axis('off')
#     xlim = ax.get_xlim()
#     ylim = ax.get_ylim()
#     # Обучаем оцениватель
#     model.fit(X, y)
#     xx, yy = np.meshgrid(np.linspace(*xlim, num=200),np.linspace(*ylim, num=200))
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
#     # Создаем цветной график с результатами
#     n_classes = len(np.unique(y))
#     contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5,
#                            cmap=cmap, clim=(y.min(), y.max()), zorder=1)
#     ax.set(xlim=xlim, ylim=ylim)
    # plt.show()
# visualize_classifier(DecisionTreeClassifier(), X, y)

# Деревья принятия решений и переобучение
# Ансамбли оценивателей: случайные леса
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
# tree = DecisionTreeClassifier()
# bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8,random_state=1)
# bag.fit(X, y)
# visualize_classifier(bag, X, y)
# plt.show()

# Ансамбль случайных деревьев принятия решений
from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, random_state=0)
# visualize_classifier(model, X, y)
# plt.show()

# Регрессия с помощью случайных лесов
# rng = np.random.RandomState(42)
# x = 10 * rng.rand(200)
# def model(x, sigma=0.3):
#     fast_oscillation = np.sin(5 * x)
#     slow_oscillation = np.sin(0.5 * x)
#     noise = sigma * rng.randn(len(x))
#     return slow_oscillation + fast_oscillation + noise
# y = model(x)
# plt.errorbar(x, y, 0.3, fmt='o')
# plt.show()

# Аппроксимация данных моделью на основе случайного леса
# from sklearn.ensemble import RandomForestRegressor
# forest = RandomForestRegressor(200)
# forest.fit(x[:, None], y)
# xfit = np.linspace(0, 10, 1000)
# yfit = forest.predict(xfit[:, None])
# ytrue = model(xfit, sigma=0)
# plt.errorbar(x, y, 0.3, fmt='o', alpha=0.5)
# plt.plot(xfit, yfit, '-r')
# plt.plot(xfit, ytrue, '-k', alpha=0.5)
# plt.show()  # настоящая модель показана гладкой, а модель на основе случайного леса — «рваной» кривой.

# Пример: использование случайного леса для классификации цифр
# from sklearn.datasets import load_digits
# digits = load_digits()
# digits.keys()
# Настройки рисунка
# fig = plt.figure(figsize=(6, 6)) # размер рисунка в дюймах
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05,wspace=0.05)
# Рисуем цифры: размер каждого изображения 8 x 8 пикселов
# for i in range(64):
#     ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
#     ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
#     # Маркируем изображение целевыми значениями
#     ax.text(0, 7, str(digits.target[i]))
#     plt.show()
# from sklearn.model_selection import train_test_split
# Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target,random_state=0)
# model = RandomForestClassifier(n_estimators=1000)
# model.fit(Xtrain, ytrain)
# ypred = model.predict(Xtest)
# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(ytest, ypred)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()  # Матрица различий для классификации цифр с помощью случайных лесов
