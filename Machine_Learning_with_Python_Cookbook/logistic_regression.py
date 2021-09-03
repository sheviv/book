# ch_16_Логистическая регрессия

# Тренировка бинарного классификатора(натренировать классификационную модель)
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# Загрузить данные только с двумя классами
# iris = datasets.load_iris()
# features = iris.data[:100, :]
# target = iris.target[:100]
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект логистической регрессии
# logistic_regression = LogisticRegression(random_state=0)
# Натренировать модель
# model = logistic_regression.fit(features_standardized, target)

# Тренировка мультикпассового классификатора(более двух классов, требуется натренировать классификационную модель)
# Натренировать логистическую регрессию используя методы "один против остальных" либо полиномиальные методы
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект логистической регрессии по методу "один против остальных"
# logistic_regression = LogisticRegression(random_state=0, multi_class="ovr")
# Натренировать модель
# model = logistic_regression.fit(features_standardized, target)

# Снижение дисперсии с помощью регуляризации
# Настроить гиперпараметр силы регуляризации C
from sklearn.linear_model import LogisticRegressionCV
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект-классификатор на основе дерева принятия решений
# logistic_regression = LogisticRegressionCV(penalty='12', Cs=10, random_state=0, n_jobs=-1)
# Натренировать модель
# model = logistic_regression.fit(features_standardized, target)

# Тренировка классификатора на очень крупных данных(натренировать простую классификационную модель на крупном наборе)
# с помощью стохастического среднеградиентного(stochastic average gradient, SAG) решателя
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект логистической регрессии
# logistic_regression = LogisticRegression(random_state=0, solver="sag")
# Натренировать модель
# model = logistic_regression.fit(features_standardized, target)

# Обработка несбалансированных классов(натренировать простую классификационную модель)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Сделать класс сильно несбалансированным, удалив первые 40 наблюдений
# features = features[40:,:]
# target = target[40:]
# Создать вектор целей, указав либо класс 0, либо 1
# target = np.where((target == 0), 0, 1)
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект-классификатор дерева принятия решений(class_weight - взвешивания классов, сбалансированное сочетание каждого класса)
# logistic_regression = LogisticRegression(random_state=0, class_weight="balanced")
# Натренировать модель
# model = logistic_regression.fit(features_standardized, target)