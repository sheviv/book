# ch_19_Кластеризация

# Кластеризация с помощью к средних(сгруппировать наблюдения в K групп)
# Использовать кластеризацию методом K средних
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# Стандартизировать признаки
# scaler = StandardScaler()
# features_std = scaler.fit_transform(features)
# Создать объект k средних
# cluster = KMeans(n_clusters=3, random_state=0, n_jobs=-1)
# Натренировать модель
# model = cluster.fit(features_std)

# Ускорение кластеризации методом K средних(сгруппировать наблюдения в K групп, но алгоритм K средних занимает много времени)
# Использовать мини-пакетный алгоритм K средних(в мини-пакетном алгоритме вычислительно затратный шаг выполняется только
# на случайной выборке наблюдений в отличие от всех наблюдений, используемых в обычном алгоритме)
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# Стандартизировать признаки
# scaler = StandardScaler()
# features_std = scaler.fit_transform(features)
# Создать объект k средних
# cluster = MiniBatchKMeans(n_clusters=3, random_state=0, batch_size=100)
# Натренировать модель
# model = cluster.fit(features_std)

# Кластеризация методом сдвига K среднему(сгруппировать наблюдения без учета количества кластеров или их формы)
# Гравитация - точки начинают сгущаться к ближайшим скоплениям, образуя кучи
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# Стандартизировать признаки
# scaler = StandardScaler()
# features_std = scaler.fit_transform(features)
# Создать объект кластеризации методом сдвига к среднему
# cluster = MeanShift(n_jobs=-1)
# Натренировать модель
# model = cluster.fit(features_std)

# сгруппировать наблюдения в кластеры высокой плотности
# Кластеризаця методом DBSCAN(Density-Based Spatial Clustering of Applications with Noise, плотностной алгоритм пространственной кластеризации с присутствием шума)
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# Стандартизировать признаки
# scaler = StandardScaler()
# features_std = scaler.fit_transform(features)
# Создать объект плотностной кластеризации dbscan
# cluster = DBSCAN(n_jobs=-1)
# Натренировать модель
# model = cluster.fit(features_std)
# eps — макс расстояние от наблюдения, чтобы считать другое наблюдение его соседом
# min_sampies — мин число наблюдений, находящихся на расстоянии менее eps от наблюдения,
# для того чтобы его можно было считать ключевым наблюдением
# metric — метрический показатель расстояния, используемый параметром eps, например minkowski или euclidean

# Кластеризация методом иерархического слияния(сгруппировать наблюдения, используя иерархию кластеров)
# Использовать агломеративную(алгоритм, который выполняет кластеризацию иерархически) кластеризацию
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# Стандартизировать признаки
# scaler = StandardScaler()
# features_std = scaler.fit_transform(features)
# Создать объект агломеративной кластеризации
# cluster = AgglomerativeClustering(n_clusters=3)
# Натренировать модель
# model = cluster.fit(features_std)
