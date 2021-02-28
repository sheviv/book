import numpy as np

# Плохой способ проверки модели
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(X, y)
# y_model = model.predict(X)
from sklearn.metrics import accuracy_score
# ac = accuracy_score(y, y_model)  # 1.0 - обучение и оценка модели выполняются на одних и тех же данных

# Хороший способ проверки модели: отложенные данные
from sklearn.model_selection import train_test_split
# Разделяем данные: по 50% в каждом из наборов
# X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
# Обучаем модель на одном из наборов данных
# model = KNeighborsClassifier(n_neighbors=1)
# model.fit(X1, y1)
# Оцениваем работу модели на другом наборе
# y2_model = model.predict(X2)
# ac = accuracy_score(y2, y2_model)  # 90.6

# Перекрестная проверка модели(cross-validation) выполнение последовательности аппроксимаций(каждое подмножество данных
# используется в качестве обучающей последовательности, так и проверочного набора)
model = KNeighborsClassifier(n_neighbors=1)
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)
model.fit(X1, y1)
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)  # — две оценки точности(двухблочной перекрестной проверкой)
# пятиблочной перекрестной проверки
from sklearn.model_selection import cross_val_score
# df = cross_val_score(model, X, y, cv=5)
# перекрестной проверки по отдельным объектам(leave-one-out cross-validation — «перекрестная проверка по всем без одного»)
from sklearn.model_selection import LeaveOneOut
# scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))

# Выбор оптимальной модели
# если наш оцениватель показывает недостаточно хорошие результаты:
#     - использовать более сложную/гибкую модель
#     - применять менее сложную/гибкую модель
#     - собрать больше выборок для обучения
#     - собрать больше данных для добавления новых признаков к каждой заданной выборке

# Поведение кривой обучения(график оценок для обучения/проверки с учетом размера обучающей последовательности):
#     - Модель заданной сложности окажется переобученной на слишком маленьком наборе данных - оценка эффективности
#     для обучения будет относительно высокой, а оценка эффективности для проверки — относительно низкой
#     - Модель заданной сложности окажется недообученной на слишком большом наборе данных - оценка эффективности
#     для обучения будет снижаться, а оценка эффективности для проверки — повышаться по мере роста размера набора данных.
#     - Модель никогда, разве что случайно, не покажет на проверочном наборе лучший результат,
#     чем на обучающей последовательности - кривые будут сближаться, но никогда не пересекутся.

# Кривые обучения в библиотеке Scikit-Learn.
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import learning_curve
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
# def PolynomialRegression(degree=2, **kwargs):
#     return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
# for i, degree in enumerate([2, 9]):
#     N, train_lc, val_lc = learning_curve(PolynomialRegression(degree), X, y, cv=7,train_sizes=np.linspace(0.3, 1, 25))
#     ax[i].plot(N, np.mean(train_lc, 1), color='blue', label='training score')
#     ax[i].plot(N, np.mean(val_lc, 1), color='red', label='validation score')
#     ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray', linestyle='dashed')
#     ax[i].set_ylim(0, 1)
#     ax[i].set_xlim(N[0], N[-1])
#     ax[i].set_xlabel('training size') # Размерность обучения
#     ax[i].set_ylabel('score')
#     ax[i].set_title('degree = {0}'.format(degree), size=14)
#     ax[i].legend(loc='best')
#     plt.show()

