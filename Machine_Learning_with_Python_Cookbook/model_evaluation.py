# ch_11_Оценивание моделей

# оценить, насколько хорошо модель будет работать в реальном мире
# Создать конвейер, который предварительно обрабатывает данные, тренирует модель и оценивает ее перекрестной проверкой
# Загрузить библиотеки
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
# Загрузить набор данных рукописных цифр
# digits = datasets.load_digits()
# Создать матрицу признаков
# features = digits.data
# Создать вектор целей
# target = digits.target
# Создать стандартизатор
# standardizer = StandardScaler()
# Создать объект логистической регрессии
# logit = LogisticRegression()
# Создать конвейер, который стандартизирует,
# затем выполняет логистическую регрессию
# pipeline = make_pipeline(standardizer, logit)
# Создать k-блочную перекрестную проверку
# kf = KFold(n_splits=10, shuffle=True, random_state=1)
# Выполнить k-блочную перекрестную проверку
# cv_results = cross_val_score(pipeline,  # Конвейер
#                              features,  # Матрица признаков
#                              target,  # Вектор целей
#                              cv=kf,  # Метод перекрестной проверки
#                              scoring="accuracy",  # Функция потери
#                              n_jobs=-1)  # Использовать все ядра CPU
# Вычислить среднее значение
# cv_results.mean()

# регрессионная модель для сравнения с вашей моделью
# объект класса DummyRegressor(sklearn) - простая модель фиктивной регрессии для использования в качестве ориентира
# Загрузить библиотеки
from sklearn.datasets import load_boston
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
# Загрузить данные
# boston = load_boston()
# Создать матрицу признаков и вектор целей
# features, target = boston.data, boston.target
# Разбить на тренировочный и тестовый наборы
# features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)
# Создать фиктивный регрессор
# dummy = DummyRegressor(strategy='mean')
# "Натренировать" фиктивный регрессор
# dummy.fit(features_train, target_train)
# Получить оценку коэффициента детерминации (R-squared)
# dummy.score(features_test, target_test)
# Для сравнения тренируем модель и вычисляем оценку результативности
from sklearn.linear_model import LinearRegression
# Натренировать простую линейно-регрессионную модель
# ols = LinearRegression()
# ols.fit(features_train, target_train)
# Получить оценку коэффициента детерминации (R-squared)
# ols.score(features_test, target_test)
# и
# Создать фиктивный регрессор, который предсказывает 20 для всех наблюдений
# elf = DummyRegressor(strategy='constant', constant=20)
# elf.fit(features_train, target_train)
# Вычислить оценку
# elf.score(features_test, target_test)

# простой базовый классификатор для сравнения с моделью
# Использовать класс Dummyciassifier(sklearn)
# Загрузить библиотеки
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
# Загрузить данные
# iris = load_iris()
# Создать матрицу признаков и вектор целей
# features, target = iris.data, iris.target
# Разбить на тренировочный и тестовый наборы
# features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=0)
# Создать фиктивный классификатор
# dummy = DummyClassifier(strategy='uniform', random_state=1)
# "Натренировать" модель
# dummy.fit(features_train, target_train)
# Получить оценку точности
# dummy.score(features_test, target_test)
# Путем сопоставления базового классификатора с натренированным классификатором видны улучшения
from sklearn.ensemble import RandomForestClassifier
# Создать классификатор случайного леса
# classifier = RandomForestClassifier()
# Натренировать модель
# classifier.fit(features_train, target_train)
# Получить оценку точности
# classifier.score(features_test, target_test)

# оценить качество натренированной классификационной модели
# Применить метод cross_val_score(sklearn) для перекрестной проверки,
# используя параметр scoring для определения одного из нескольких метрических показателей
# результативности(точность, прецизионность, полноту и оценку F1)
# измерить точность в трехблочной(количество блоков) перекрестной проверке(scoring="accuracy")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
# Сгенерировать матрицу признаков и вектор целей
# X, y = make_classification(n_samples=10000,n_features=3,n_informative=3,n_redundant=0,n_classes=2,random_state=1)
# Создать объект логистической регрессии
# logit = LogisticRegression()
# Перекрестно проверить модель, используя показатель точности
# cross_val_score(logit, X, y, scoring="accuracy") - доля наблюдений, предсказанных правильно
# и
# Перекрестно проверить модель, используя показатель прецизионности
# cross_val_score(logit, X, y, scoring="precision")
# и
# Перекрестно проверить модель, используя показатель полноты
# cross_val_score(logit, X, у, scoring="recall")
# и
# Перекрестно проверить модель, используя показатель F1(среднее гармоническое)
# cross_val_score(logit, X, у, scoring="f1")
# если есть истинные значения у и предсказанные значения у^, вычислить метрические показатели(точность, полнота)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Разбить на тренировочный и тестовый наборы
# X_train, X_test, y_train, y_test = train_test_split(X,У,test_size=0.1,random_state=1)
# Предсказать значения для тренировочного вектора целей
# y_hat = logit.fit(X_train, y_train).predict(X_test)
# Вычислить точность
# accuracy_score(y_test, y_hat)

# оценить двоичный классификатор и вероятностные пороги
# для истинно - и ложноположительных исходов на каждом пороге используют метод roc_curve и вывести их на график
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
# Создать матрицу признаков и вектор целей
# features, target = make_classification(n_samples=10000, n_features=10, n_classes=2, n_informative=3, random_state=3)
# Разбить на тренировочный и тестовый наборы
# features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.1, random_state=1)
# Создать логистический регрессионный классификатор
# logit = LogisticRegression()
# Натренировать модель
# logit.fit(features_train, target_train)
# Получить предсказанные вероятности
# target_probabilities = logit.predict_proba(features_test)[:, 1]
# Создать доли истинно- и ложноположительных исходов
# false_positive_rate, true_positive_rate, threshold = roc_curve(target_test, target_probabilities)
# Построить график кривой ROC
# plt.title("Кривая ROC")
# plt.plot(false_positive_rate, true_positive_rate)
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0], c=".7")
# plt.plot([1, 1], c=".7")
# plt.ylabel("Доля истинноположительных исходов")
# plt.xlabel("Доля ложноположительных исходов")
# plt.show()
# Получить предсказанные вероятности
# logit.predict_proba(features_test)[0:1]
# Вычислить площадь под кривой ROC AUC
# roc_auc_score(target_test, target_probabilities)

# оценить результативность модели, которая предсказывает три класса или более
# Использовать перекрестную проверку с оценочным метрическим показателем, способным справляться с более чем двумя классами
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
# Сгенерировать матрицу признаков и вектор целей
# features, target = make_classification(n_samples = 10000,n_features = 3,n_informative = 3,n_redundant = 0,n_classes = 3,
# random_state = 1)
# Создать объект логистической регрессии
# logit = LogisticRegression()
# Перекрестно проверить модель, используя показатель точности
# cross_val_score(logit, features, target, scoring='accuracy')
# и
# Перекрестно проверить модель используя макроусредненную оценку F1
# cross_val_score(logit, features, target, scoring='f1_macro')
# macro — рассчитать среднее метрических оценок для каждого класса, взвешивая каждый класс одинаково
# weighted — рассчитать среднее метрических оценок для каждого класса, взвешивая каждый класс пропорционально его размеру в данных
# micro — рассчитать среднее метрических оценок для каждой комбинации "класс — наблюдение"

# визуально сопоставить качество модели предсказанных классов и истинных классов тестовых данных
# Использовать матрицу ошибок(несоответствий), которая сравнивает предсказанные и истинные классы
# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
# Загрузить данные
# iris = datasets.load_iris()
# Создать матрицу признаков
# features = iris.data
# Создать вектор целей
# target = iris.target
# Создать список имен целевых классов
# class_names = iris.target_names
# Создать тренировочный и тестовый наборы
# features_train, features_test, target_train, target_test = train_test_split(features, target, random_state=1)
# Создать объект логистической регрессии
# classifier = LogisticRegression()
# Натренировать модель и сделать предсказания
# target_predicted = classifier.fit(features_train, target_train).predict(features_test)
# Создать матрицу ошибок
# matrix = confusion_matrix(target_test, target_predicted)
# Создать фрейм данных pandas
# dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Создать тепловую карту
# sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
# pit.title("Матрица ошибок")
# pit.tight_layout()
# pit.ylabel("Истинный класс")
# pit.xlabel("Предсказанный класс")
# pit.show()

# оценить результативность регрессионной модели
# Использовать среднеквадратическую ошибку(mean squared error, MSE)
# Загрузить библиотеки
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
# Сгенерировать матрицу признаков, вектор целей
# features, target = make_regression(n_samples=100,n_features=3,n_informative=3,n_targets=1,noise=50,coef=False,random_state=1)
# Создать объект линейной регрессии
# ols = LinearRegression()
# Перекрестно проверить линейную регрессию, используя (отрицательный) показатель MSE
# cross_val_score(ols, features, target, scoring='neg_mean_squared_error')
# или
# метрический показатель регрессии является коэффициент детерминации R2
# Перекрестно проверить линейную регрессию, используя показатель R-квадрат
# cross_val_score(ols, features, target, scoring='r2')

# насколько хорош неконтролируемо обучающийся алгоритм(кластеризация данных)
# оценка кластеризации с использованием силуэтных коэффициентов, которые оценивают качество кластеров
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Сгенерировать матрицу признаков
# features, _ = make_blobs(n_samples=1000,n_features=10,centers=2,cluster_std=0.5,shuffle=True,random_state=1)
# Кластеризовать данные, используя алгоритм к средних, чтобы предсказать классы
# model = KMeans(n_clusters=2,random_state=1).fit(features)
# Получить предсказанные классы
# target_predicted = model.labels_
# Оценить модель
# silhouette_score(features, target_predicted)

# оценить модель с помощью созданного метрического показателя
# Создать метрический показатель как функцию и конвертировать его в оценочную функцию, используя make_scorer(sklearn)
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
# Сгенерировать матрицу признаков и вектор целей
# features, target = make_regression(n_samples=100,n_features=3,random_state=1)
# Создать тренировочный и тестовый наборы
# features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.10, random_state=1)
# Создать собственный метрический показатель
# def custom_metric(target_test, target_predicted):
    # Вычислить оценочный показатель r-квадрат
    # r2 = r2_score(target_test, target_predicted)
    # Вернуть оценочный показатель r-квадрат
    # return r2
# Создать оценочную функцию и установить, что чем выше оценки, тем они лучше
# score = make_scorer(custom_metric, greater_is_better=True)
# Создать объект гребневой регрессии
# classifier = Ridge()
# Натренировать гребневую регрессионную модель
# model = classifier.fit(features_train, target_train)
# Применить собственную оценочную функцию
# score(model, features_test, target_test)
# и
# Предсказать значения
# target_predicted = model.predict(features_test)
# Вычислить оценочный показатель r-квадрат
# r2_score(target_test, target_predicted)

# оценить влияния количества наблюдений в тренировочном наборе на некоторый метрический показатель(точность,F1 и т.д.)
# Построить график кривой заучивания(кривая освоения) - используются для определения того, выиграют ли обучающиеся
# алгоритмы от сбора дополнительных тренировочных данных.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
# Загрузить данные
# digits = load_digits()
# Создать матрицу признаков и вектор целей
# features, target = digits.data, digits.target
# Создать перекрестно-проверочные тренировочные и тестовые оценки для разных размеров тренировочного набора
# train_sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(), # Классификатор
#                                                         features, # Матрица признаков
#                                                         target, # Вектор целей
#                                                         cv=10, # Количество блоков
#                                                         scoring='accuracy', # Показатель результативности
#                                                         n_jobs=-1, # Использовать все ядра CPU
#                                                         train_sizes=np.linspace(0.01, 1.0, 50))  # Размеры 50 тренировочных наборов
# Создать средние и стандартные отклонения оценок тренировочного набора
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# Создать средние и стандартные отклонения оценок тестового набора
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# Нанести линии
# plt.plot(train_sizes, train_mean, '--', color="#111111",label="Тренировочная оценка")
# plt.plot(train_sizes, test_mean, color="#111111",label="Перекрестно-проверочная оценка")
# Нанести полосы
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
# Построить график
# plt.title("Кривая заучивания")
# plt.xlabel("Размер тренировочного набора")
# plt.ylabel("Оценка точности")
# plt.legend(loc="best")
# plt.tight_layout()
# plt.show()

# как результативность модели изменяется по мере изменения значений некоторого гиперпараметра
# Построить график валидационной кривой
# import matplotlib.pyplot as pit
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import validation_curve
# Загрузить данные
# digits = load_digits()
# Создать матрицу признаков и вектор целей
# features, target = digits.data, digits.target
# Создать диапазон значений для параметра
# param_range = np.arange(1, 250, 2)
# Вычислить точность на тренировочном и тестовом наборах, используя диапазон значений параметра
# train_scores, test_scores = validation_curve(RandomForestClassifier(), # Классификатор
#                                              features,  # Матрица признаков
#                                              target,  # Вектор целей
#                                              param_name="n_estimators",  # Исследуемый гиперпараметр/имя варьируемого гиперпараметра
#                                              param_range=param_range,  # Диапазон значений гиперпараметра/используемое значение гиперпараметра
#                                              cv=3,  # Количество блоков
#                                              scoring="accuracy",  # Показатель результативности/оценочный метрический показатель для оценивания модели
#                                              n_jobs=-1)  # Использовать все ядра CPU
# Вычислить среднее и стандартное отклонение для оценок тренировочного набор
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# Вычислить среднее и стандартное отклонение для оценок тестового набора
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# Построить график средних оценок точности для тренировочного и тестового наборов
# pit.plot(param_range, train_mean, color="black", label="Тренировочная оценка")
# pit.plot(param_range, test_mean, color="dimgrey", label="Перекрестно-проверочная оценка")
# Нанести полосы точности для тренировочного и тестового наборов
# plt.fill_between(param_range, train_mean - train_std,train_mean + train_std, color="gray")
# plt.fill_between(param_range, test_mean - test_std,test_mean + test_std, color="gainsboro")
# Создать график
# pit.title("Валидационная кривая со случайным лесом")
# pit.xlabel("Количество деревьев")
# pit.ylabel("Оценка точности")
# pit.tight_layout()
# pit.legend(loc="best")
# pit.show()