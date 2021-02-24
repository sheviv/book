# ch_14_Деревья и леса

# Тренировка классификационного дерева принятия решений(натренировать классификатор, используя дерево принятия решений)
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор дерева принятия решений
# decisiontree = DecisionTreeClassifier(random_state=0)
# Натренировать модель
# model = decisiontree.fit(features, target)
# Сконструировать новое наблюдение
# observation = [[5, 4, 3, 2]]
# Предсказать класс наблюдения
# model.predict(observation)
# Взглянуть на предсказанные вероятности трех классов
# model.predict_proba(observation)
# или
# если требуется использовать другую меру разнородности
# Создать объект-классификатор дерева принятия решений, используя энтропию
# decisiontree_entropy = DecisionTreeClassifier(criterion='entropy', random_state=0)
# Натренировать модель
# model_entropy = decisiontree_entropy.fit(features, target)

# Тренировка регрессионного дерева принятия решений(натренировать регрессионную модель с помощью дерева принятия решений)
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
# Загрузить данные всего с двумя признаками
# boston = datasets.load_boston()
# features = boston.data[:, 0:2]
# target = boston.target
# Создать объект-классификатор дерева принятия решений
# decisiontree = DecisionTreeRegressor(random_state=0)
# Натренировать модель
# model = decisiontree.fit(features, target)
# и
# Создать объект-классификатор дерева принятия решений, используя энтропию
# decisiontree_mae = DecisionTreeRegressor(criterion="mae", random_state=0)
# Натренировать модель
# model_mae = decisiontreejnae.fit(features, target)

# Визуализация модели дерева принятия решений(визуализировать модель, обучающимся алгоритмом дерева принятия решений)
# Экспортировать модель дерева принятия решений в формат DOT, затем визуализировать
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image
from sklearn import tree
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор дерева принятия решений
# decisiontree = DecisionTreeClassifier(random_state=0)
# Натренировать модель
# model = decisiontree.fit(features, target)
# Создать данные в формате DOT
# dot_data = tree.export_graphviz(decisiontree, out_file=None,
#                                 feature_names=iris.feature_names, class_names=iris.target_names)
# Начертить граф
# graph = pydotplus.graph_from_dot_data(dot_data)
# Показать граф
# Image(graph.create_png())
# Создать PDF
# graph.write_pdf("iris.pdf")
# Создать PNG
# graph.write_png("iris.png")

# Тренировка классификационного случайного леса
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор случайного леса
# randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Натренировать модель
# model = randomforest.fit(features, target)

# Тренировка регрессионного случайного леса
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
# Загрузить данные только с двумя признаками
# boston = datasets.load_boston()
# features = boston.data[:,0:2]
# target = boston.target
# Создать объект-классификатор случайного леса
# randomforest = RandomForestRegressor(random_state=0, n_jobs=-1)
# Натренировать модель
# model = randomforest.fit(features, target)
# и
# max_features - макс кол-во признаков для рассмотрения на каждом узле(sqrt(p) признаков,р — это общее кол-во признаков
# bootstrap - устанавливает(выполнять ли выборку с возвратом или нет(True)
# n_estimators - количество конструируемых деревьев принятия решений(10)

# Идентификация важных признаков в случайных лесах(какие признаки наиболее важны в модели случайного леса)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор случайного леса
# randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Натренировать модель
# model = randomforest.fit(features, target)
# Вычислить важности признаков
# importances = model.feature_importances_
# Отсортировать важности признаков в нисходящем порядке
# indices = np.argsort(importances)[::-1]
# Перераспределить имена признаков, чтобы они совпадали с отсортированными важностями признаков
# names = [iris.feature_names[i] for i in indices]
# Создать график
# plt.figure()
# Создать заголовок графика
# plt.title("Важности признаков")
# Добавить столбики
# plt.bar(range(features.shape[1]), importances[indices])
# Добавить имена признаков как метки оси х
# plt.xticks(range(features.shape[1]), names, rotation=90)
# Показать график
# plt.show()

# Отбор важных признаков в случайных лесах
# Идентифицировать важные признаки и перетренировать модель с использованием только наиболее важных признаков
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.feature_selection import SelectFromModel
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор случайного леса
# randomforest = RandomForestClassifier(random_state=0, n_jobs=-1)
# Создать объект, который отбирает признаки с важностью, большей или равной порогу
# selector = SelectFromModel(randomforest, threshold=.3)
# Выполнить подгонку новой матрицы признаков, используя селектор
# features_important = selector.fit_transform(features, target)
# Натренировать случайный лес, используя наиболее важные признаки
# model = randomforest.fit(features_important, target)

# Обработка несбалансированных классов
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Сделать класс сильно несбалансированным, удалив первые 40 наблюдений
# features = features[40:,:]
# target = target[40:]
# Создать вектор целей, назначив классам значение 0 либо 1
# target = np.where((target == 0), 0, 1)
# Создать объект-классификатор случайного леса
# randomforest = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced")
# Натренировать модель
# model = randomforest.fit(features, target)

# Управление размером дерева(определить структуру и размер дерева принятия решений)
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор дерева принятия решений
# decisiontree = DecisionTreeClassifier(random_state=0,max_depth=None,min_samples_split=2,min_samples_leaf=1,
#                                       min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0)
# Натренировать модель
# model = decisiontree.fit(features, target)
# max_depth — макс. глубина дерева(None - дерево выращивается пока все листья не станут однородными,
# число - дерево практически "обрезается" до этой глубины
# min_sampies_spiit — минимальное кол-во наблюдений в узле прежде, чем он будет расщеплен.
# целое число - оно определяет голый минимум,если вещественное, то минимум равен проценту от общего кол-ва наблюдений
# min_sampies_ieaf — мин кол-во наблюдений, которое должно находиться в листе(аргументы, что и min_sampies_spiit)
# max_leaf_nodes — макс количество листьев.
# min_impurity_decrease — мин требуемое уменьшение разнородности прежде, чем расщепление выполнится

# Улучшение результативности с помощью бустинга(ряд слабых моделей(чаще всего это - пень))
# Натренировать бустированную модель
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор дерева принятия решений на основе алгоритма adaboost
# adaboost = AdaBoostClassifier(random_state=0)
# Натренировать модель
# model = adaboost.fit(features, target)

# Оценивание случайных лесов с помощью ошибок внепакетных наблюдений(оценить модель без перекрестной проверки)
# Вычислить внепакетную (out-of-bag, ООВ) оценку модели
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор случайного леса
# randomforest = RandomForestClassifier(random_state=0, n_estimators=1000, oob_score=True, n_jobs=-1)
# Натренировать модель
# model = randomforest.fit(features, target)
# Взглянуть на внепакетную ошибку
# randomforest.oob_score_