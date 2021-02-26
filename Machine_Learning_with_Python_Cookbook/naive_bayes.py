# ch_18_Наивный Байес

# Тренировка классификатора для непрерывных признаков(даны непрерывные признаки, требуется натренировать наивный байесов классификатор)
# Использовать гауссов наивный байесов классификатор(лучше использовать, когда все признаки непрерывны)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект гауссова наивного Байеса
# classifer = GaussianNB()
# Натренировать модель
# model = classifer.fit(features, target)

# Тренировка классификатора для дискретных и счетных признаков(дискретные или счетные данные,
# требуется натренировать наивный байесов классификатор)
# Использовать полиномиальный наивный байесов классификатор
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# Создать текст
# text_data = np.array(['Бразилия - моя любовь. Бразилия!','Бразилия - лучше','Германия бьет обоих'])
# Создать мешок слов
# count = CountVectorizer()
# bag_of_words = count.fit_transform(text_data)
# Создать матрицу признаков
# features = bag_of_words.toarray()
# Создать вектор целей
# target = np.array([0,0,1])
# Создать объект полиномиального наивного байесова классификатора с априорными вероятностями каждого класса
# classifer = MultinomialNB(class_prior=[0.25, 0.5])
# Натренировать модель
# model = classifer.fit(features, target)

# Тренировка наивного байесова классификатора для бинарных признаков(двоичные признаковые данные,
# требуется натренировать наивный байесов классификатор)
# Использовать бернуллиев наивный байесов классификатор
import numpy as np
from sklearn.naive_bayes import BernoulliNB
# Создать три бинарных признака
# features = np.random.randint(2, size=(100, 3))
# Создать вектор бинарных целей
# target = np.random.randint(2, size=(100, 1)).ravel()
# Создать объект бернуллиева наивного Байеса с априорными вероятностями каждого класса
# classifer = BernoulliNB(class_prior=[0.25, 0.5])
# Натренировать модель
# model = classifer.fit(features, target)
# Если требуется указать равномерную априорную вероятность,указывают fit_prior=False
# model_uniform_prior = BernoulliNB(class_prior=None, fit_prior=True)

# Калибровка предсказанных вероятностей(откалибровать предсказанные вероятности из наивных байесовых классификаторов,
# чтобы они были интерпретируемыми)
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект гауссова наивного Байеса
# classifer = GaussianNB()
# Создать откалиброванную перекрестную проверку с сигмоидальной калибровкой
# classifer_sigmoid = CalibratedClassifierCV(classifer, cv=2,method='sigmoid')
# Откалибровать вероятности
# classifer_sigmoid.fit(features, target)
# Создать новое наблюдение
# new_observation = [[2.6, 2.6, 2.6, 0.4]]
# Взглянуть на откалиброванные вероятности
# classifer_sigmoid.predict_proba(new_observation)