# ch_15_K ближайших соседей

# Отыскание ближайших соседей наблюдения(K ближайших наблюдений(соседей))
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# Создать стандартизатор
# standardizer = StandardScaler()
# Стандартизировать признаки
# features_standardized = standardizer.fit_transform(features)
# Два ближайших соседа
# nearest_neighbors = NearestNeighbors(n_neighbors=2).fit(features_standardized)
# или
# Найти двух ближайших соседей на основе евклидова расстояния
# nearestneighbors_euclidean = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(features_standardized)
# Создать наблюдение
# new_observation = [1, 1, 1, 1]
# Найти расстояния и индексы ближайших соседей наблюдения
# distances, indices = nearest_neighbors.kneighbors([new_observation])
# Взглянуть на ближайших соседей
# features_standardized[indices]
# для создания матрицы, показывающей ближайших соседей каждого наблюдения
# Найти трех ближайших соседей каждого наблюдения на основе евклидова расстояния(включая себя)
# nearestneighbors_euclidean = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(features_standardized)
# Список списков, показывающий трех ближайших соседей каждого наблюдения (включая себя)
# nearest_neighbors_with_self = nearestneighbors_euclidean.kneighbors_graph(features_standardized).toarray()
# Удалить единицы, отметив наблюдение, как ближайший сосед к себе
# for i, x in enumerate(nearest_neighbors_with_self):
#     x[i] = 0
# Взглянуть на ближайших соседей первого наблюдения
# nearest_neighbors_with_self[0]

# Создание классификационной модели к ближайших соседей
# Если набор данных не очень большой
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# Создать стандартизатор
# standardizer = StandardScaler()
# Стандартизировать признаки
# X_std = standardizer.fit_transform(X)
# Натренировать классификатор KNN c 5 соседями
# knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_std, y)
# Создать два наблюдения
# new_observations = [[0.75, 0.75, 0.75, 0.75],[1,1,1,1]]
# Предсказать класс двух наблюдений
# knn.predict(new_observations)

# Идентификация наилучшего размера окрестности(выбрать наилучшее значение для K в классификационной модели)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать стандартизатор
# standardizer = StandardScaler()
# Стандартизировать признаки
# features_standardized = standardizer.fit_transform(features)
# Создать классификатор KNN
# knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# Создать конвейер
# pipe = Pipeline([("standardizer", standardizer), ("knn", knn)])
# Создать пространство вариантов значений
# search_space = [{"knn__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}]
# Создать объект решеточного поиска
# classifier = GridSearchCV(pipe, search_space, cv=5, verbose=0).fit(features_standardized, target)
# Наилучший размер окрестности (к)
# classifier.best_estimator_.get_params()["knn__n_neighbors"]

# Создание радиусного классификатора ближайших соседей(предсказать класс на основе класса наблюдений в
# пределах определенного расстояния)
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать стандартизатор
# standardizer = StandardScaler()
# Стандартизировать признаки
# features_standardized = standardizer.fit_transform(features)
# Натренировать радиусный классификатор соседей
# rnn = RadiusNeighborsClassifier(radius=.5, n_jobs=-1).fit(features_standardized, target)
# Создать наблюдение
# new_observations = [[1, 1, 1, 1]]
# Предсказать класс наблюдения
# rnn.predict(new_observations)