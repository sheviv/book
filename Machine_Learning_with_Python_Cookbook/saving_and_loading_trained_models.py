# ch_21_Сохранение и загрузка натренированных моделей

# Сохранение и загрузка натренерованной модели scikit-learn
# Сохранить модель в качестве файла консервации
# Загрузить библиотеки
from joblib import load, dump
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
# Загрузить данные
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект-классификатор случайных лесов
# classifer = RandomForestClassifier()
# Натренировать модель
# model = classifer.fit(features, target)
# Сохранить модель в качестве файла консервации
# dump(model, "model.pkl")
# Загрузить модель из файла
# classifer = load("model.pkl")
# И использовать ее для выполнения предсказаний:
# Создать новое наблюдение
# new_observation = [[5.2, 3.2, 1.1, 0.1]]
# Предсказать класс наблюдения
# classifer.predict(new_observation)
# и
# полезно включить используемую в модели версию библиотеки scikit-lean в имя файла
# Импортировать библиотеку
import sklearn
# Получить версию библиотеки scikit-learn
# scikit_version = joblib.__version__
# Сохранить модель в качестве файла консервации
# joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))

# Сохранение и загрузка модели Keras(натренированная модель Keras, и требуется ее сохранить и загрузить в другом месте)
# Сохранить модель в файле формата HDF5
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.models import load_model
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные отзывов о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# train_features = tokenizer.sequences_to_matrix(train_data, mode="binary")
# test_features = tokenizer.sequences_to_matrix(test_data, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16,
#                          activation="relu",
#                          input_shape=(number_of_features,)))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Train neural network
# history = network.fit(train_features, # Признаки
#                       train_target, # Вектор целей
#                       epochs=3, # Количество эпох
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(test_features, test_target)) # Тестовые данные
# Сохранить нейронную сеть
# network.save("model.h5")
# и
# загрузить модель в другом приложении либо для дополнительной тренировки
# Загрузить нейронную сеть
# network = load_model("model.h5")
