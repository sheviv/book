# ch_20_Нейронные сети

# Предобработка данных для нейронных сетей
# Стандартизировать каждый признак, используя класс standardscaier
from sklearn import preprocessing
# import numpy as np
# Создать признак
# features = np.array([[-100.1, 3240.1],
#                      [-200.2, -234.1],
#                      [5000.5, 150.1],
#                      [6000.6, -125.1],
#                      [9000.9, -673.1]])
# Создать шкалировщик
# scaler = preprocessing.StandardScaler()
# Преобразовать признак
# features_standardized = scaler.fit_transform(features)

# Проектирование нейронной сети(спроектировать нейронную сеть)
from keras import models
from keras import layers
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy",  # Перекрестная энтропия
#                 optimizer="rmsprop",  # Распространение среднеквадратической ошибки
#                 metrics=["accuracy"])  # Точностный показатель результативности

# Тренировка бинарного классификатора(натренировать бинарно-классификационную нейронную сеть)
# Keras для построения нейронной сети прямого распространения и ее тренировки с помощью метода fit
# import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
# Задать начальное значение для генератора псевдослучайных чисел
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные с отзывами о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu", input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy",  # Перекрестная энтропия
#                 optimizer="rmsprop",  # Распространение среднеквадратической ошибки
#                 metrics=["accuracy"])  # Точностный показатель результативности
# Натренировать нейронную сеть
# history = network.fit(features_train,  # Признаки
#                       target_train,  # Вектор целей
#                       epochs=3,  # Количество эпох
#                       verbose=1,  # Печатать описание после каждой эпохи
#                       batch_size=100,  # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test))  # Тестовые данные

# Тренировка мультиклассового классификатора(натренировать мультиклассовую классификационную нейронную сеть)
# Использовать библиотеку Keras для построения нейронной сети прямого распространения с выходным слоем с активационными функциями softmax
import numpy as np
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 5000
# Загрузить признаковые и целевые данные
# data = reuters.load_data(num_words=number_of_features)
# (data_train, target_vector_train), (data_test, target_vector_test) = data
# Конвертировать признаковые данные в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Преобразовать вектор целей в кодировке с одним активным состоянием, чтобы создать матрицу целей
# target_train = to_categorical(target_vector_train)
# target_test = to_categorical(target_vector_test)
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=100, activation="relu", input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=100, activation="relu"))
# Добавить полносвязный слой с активационной функцией softmax
# network.add(layers.Dense(units=46, activation="softmax"))
# Скомпилировать нейронную сеть
# network.compile(loss="categorical_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Цель
#                       epochs=3, # Три эпохи
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные

# Тренировка регрессора(натренировать регрессионную нейронную сеть)
# Keras для построения нейронной сети прямого распространения с одним выходным блоком и без активационной функции
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Сгенерировать матрицу признаков и вектор целей
# features, target = make_regression(n_samples = 10000,n_features = 3,n_informative = 3,
#                                    n_targets = 1,noise = 0.0,random_state = 0)
# Разделить данные на тренировочный и тестовый наборы
# features_train, features_test, target_train, target_test = train_test_split(features, target,
#                                                                             test_size=0.33, random_state=0)
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=32,activation="relu",input_shape=(features_train.shape[1],)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=32, activation="relu"))
# Добавить полносвязный слой без активационной функции
# network.add(layers.Dense(units=1))
# Скомпилировать нейронную сеть
# network.compile(loss="mse", # Среднеквадратическая ошибка
#                 optimizer="RMSprop", # Оптимизационный алгоритм
#                 metrics=["mse"]) # Среднеквадратическая ошибка
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Вектор целей
#                       epochs=10, # Количество эпох
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные

# Выполнение предсказаний
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 10000
# Загрузить данные и вектор целей из набора данных IMDB о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные IMDB о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16,
#                          activation="relu",
#                          input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Вектор целей
#                       epochs=3, # Количество эпох
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные
# Предсказать классы тестового набора
# predicted_target = network.predict(features_test)

# Визуализация истории процесса тренировки(найти "золотую середину" в потере и/или оценке точности нейронной сети)
# matplotlib - чтобы визуализировать потерю на тестовом и тренировочном наборах данных по каждой эпохе
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
import matplotlib.pyplot as plt
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 10000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные отзывов о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвяэный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16,
#                          activation="relu",
#                          input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Цели
#                       epochs=15, # Количество эпох
#                       verbose=0, # Вывода нет
#                       batch_size=1000, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные
# Получить истории потерь на тренировочных и тестовых данных
# training_loss = history.history["loss"]
# test_loss = history.history["val_loss"]
# Создать счетчик количества эпох
# epoch_count = range(1, len(training_loss) + 1)
# Визуализировать историю потери
# plt.plot(epoch_count, training_loss, "r--")
# plt.plot(epoch_count, test_loss, "b-")
# plt.legend(["Потеря на тренировке", "Потеря на тестировании"])
# plt.xlabel("Эпоха")
# plt.ylabel("Потеря")
# plt.show()
# и
# Получить истории потерь на тренировочных и тестовых данных
# training_accuracy = history.history["асе"]
# test_accuracy = history.history["val_acc"]
# plt.plot(epoch_count, training_accuracy, "r--")
# plt.plot(epoch_count, test_accuracy, "b-")
# Визуализировать историю потери
# plt.legend(["Потеря на тренировке", "Потеря на тестировании"])
# plt.xlabel("Эпоха")
# plt.уlabel("Оценка точности")
# plt.show()

# Снижение переподгонки с помощью регуляризации весов(снизить переподгонку)
# применить штраф к параметрам сети(регуляризацию весов)
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras import regularizers
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные отзывов о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16,
#                          activation="relu",
#                          kernel_regularizer=regularizers.l2(0.01),  # 0.01 - насколько штрафуем более высокие значения параметров
#                          input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16,
#                          kernel_regularizer=regularizers.l2(0.01),  # 0.01 - насколько штрафуем более высокие значения параметров
#                          activation="relu"))
# Добавить полносвязный слой с ситоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Вектор целей
#                       epochs=3, # Количество эпох
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные

# Снижение переподгонки с помощью ранней остановки(снизить переподгонку)
# остановить тренировку, когда потеря на тестовых данных перестанет уменьшаться - ранняя остановка
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Задать начальное значение ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные отзывов о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu",
#                          input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy",  # Перекрестная энтропия
#                 optimizer="rmsprop",  # Распространение CKO
#                 metrics=["accuracy"])  # Точностный показатель результативности
# Задать функции обратного вызова для ранней остановки тренировки и сохранения наилучшей достигнутой модели
# callbacks = [EarlyStopping(monitor="val_loss", patience=2),
             # отслеживание потери на тестовых данных в каждую эпоху,
             # и после того, как потеря на тестовых данных не улучшается
             # на протяжении двух эпох, тренировка прерывается.
             # ModelCheckpoint(filepath="best_model.h5", monitor="val_loss", save_best_only=True)
             # сохраняет модель в файл после каждой контрольной точки(полезно если по какой-то причине
             # многодневная тренировка прерывается)
             # ]
# Натренировать нейронную сеть
# history = network.fit(features_train,  # Признаки
#                       target_train,  # Вектор целей
#                       epochs=20,  # Количество эпох
#                       callbacks=callbacks,  # Ранняя остановка
#                       verbose=0,  # Печатать описание после каждой эпохи
#                       batch_size=100,  # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test))  # Тестовые данные

# Снижение переподгонки с помощью отсева(Dropout)
# Внести шумовую компоненту в сетевую архитектуру с помощью отсева
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные отзывов о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить отсеивающий слой для входного слоя
# network.add(layers.Dropout(0.2, input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить отсеивающий слой для предыдущего скрытого слоя
# network.add(layers.Dropout(0.5))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить отсеивающий слой для предыдущего скрытого слоя
# network.add(layers.Dropout(0.5))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Вектор целей
#                       epochs=3, # Количество эпох
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные

# Сохранение процесса тренировки модели.
# требуется возможность сохранить ход выполнения обучения, если процесс тренировки будет прерван
# функция обратного вызова(ModelCheckpoint в Keras), чтобы записывать модель на диск после каждой эпохи
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
from keras.callbacks import ModelCheckpoint
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Конвертировать данные отзывов о кинофильмах в матрицу признаков в кодировке с одним активным состоянием
# tokenizer = Tokenizer(num_words=number_of_features)
# features_train = tokenizer.sequences_to_matrix(data_train, mode="binary")
# features_test = tokenizer.sequences_to_matrix(data_test, mode="binary")
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16,
#                          activation="relu",
#                          input_shape=(number_of_features,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Задать функции обратного вызова для ранней остановки тренировки и сохранения наилучшей достигнутой модели
# checkpoint = [ModelCheckpoint(filepath="models.hdf5")]
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Вектор целей
#                       epochs=3, # Количество эпох
#                       callbacks=checkpoint, # Контрольная точка
#                       verbose=0, # Вывода нет
#                       batch_size=100, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные

# k-блочная перекрестная проверка нейронных сетей
# оценить нейронную сеть, используя k-блочную перекрестную проверку
# k-блочная перекрестная проверка нейронных сетей не является ни необходимой, ни целесообразной,
# если это необходимо, следует использовать оболочку библиотеки Keras для scikit-lean, чтобы позволить последовательным
# моделям Keras применять API библиотеки scikit-lean
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Количество признаков
# number_of_features = 100
# Сгенерировать матрицу признаков и вектор целей
# features, target = make_classification(n_samples = 10000, n_features = number_of_features,
#                                        n_informative = 3, n_redundant = 0,
#                                        n_classes = 2, weights = [.5, .5], random_state = 0)
# Создать функцию, возвращающую скомпилированную сеть
# def create_network():
    # Инициализировать нейронную сеть
    # network = models.Sequential()
    # Добавить полносвязный слой с активационной функцией ReLU
    # network.add(layers.Dense(units=16,
    #                          activation="relu",
    #                          input_shape=(number_of_features,)))
    # Добавить полносвязный слой с активационной функцией ReLU
    # network.add(layers.Dense(units=16, activation="relu"))
    # Добавить полносвязный слой с сигмоидальной активационной функцией
    # network.add(layers.Dense(units=1, activation="sigmoid"))
    # Скомпилировать нейронную сеть
    # network.compile(loss="binary_crossentropy", # Перекрестная энтропия
    #                 optimizer="rmsprop", # Распространение CKO
    #                 metrics=["accuracy"]) # Точностный показатель результативности
    # Вернуть скомпилированную сеть
    # return network
# Обернуть модель Keras, чтобы она могла использоваться библиотекой scikit-learn
# neural_network = KerasClassifier(build_fn=create_network,epochs=10,batch_size=100,verbose=0)
# Оценить нейронную сеть с помощью трехблочной перекрестной проверки
# cross_val_score(neural_network, features, target, cv=3)

# Тонкая настройка нейронных сетей(автоматически отобрать наилучшие гиперпараметры для нейронной сети)
# Объединить нейронную сеть библиотеки Keras с инструментами отбора модели библиотеки scikit-lean(GridSearchCV),
# реализующий поиск по решетке параметров с перекрестной проверкой
import numpy as np
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# # Количество признаков
# number_of_features = 100
# # Сгенерировать матрицу признаков и вектор целей
# features, target = make_classification(n_samples = 10000,n_features = number_of_features,n_informative = 3,
#                                        n_redundant = 0,n_classes = 2,weights = [.5, .5],random_state = 0)
# # Создать функцию, возвращающую скомпилированную сеть
# def create_network(optimizer="rmsprop"):
#     # Инициализировать нейронную сеть
#     network = models.Sequential()
#     # Добавить полносвязный слой с активационной функцией ReLU
#     network.add(layers.Dense(units=16,activation="relu",input_shape=(number_of_features,)))
#     # Добавить полносвязный слой с активационной функцией ReLU
#     network.add(layers.Dense(units=16, activation="relu"))
#     # Добавить полносвязный слой с сигмоидальной активационной функцией
#     network.add(layers.Dense(units=1, activation="sigmoid"))
#     # Скомпилировать нейронную сеть
#     network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                     optimizer=optimizer, # Оптимизатор
#                     metrics=["accuracy"]) # Точностный показатель результативности
#     # Вернуть скомпилированную сеть
#     return network
# # Обернуть модель Keras, чтобы она могла использоваться библиотекой scikit-learn
# neural_network = KerasClassifier(build_fn=create_network, verbose=0)
# # Создать гиперпараметрическое пространство
# epochs = [5, 10]
# batches = [5, 10, 100]
# optimizers = ["rmsprop", "adam"]
# # Создать словарь вариантов гиперпараметров
# hyperparameters = dict(optimizer=optimizers, epochs=epochs,batch_size=batches)
# # Создать объект решеточного поиска
# grid = GridSearchCV(estimator=neural_network, param_grid=hyperparameters)
# # Выполнить подгонку объекта решеточного поиска
# grid_result = grid.fit(features, target)

# Визуализация нейронных сетей(визуализировать архитектуру нейронной сети)
# Использовать функции model_to_dot либо plot_modei библиотеки Keras
from keras import models
from keras import layers
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu", input_shape=(10,)))
# Добавить полносвязный слой с активационной функцией ReLU
# network.add(layers.Dense(units=16, activation="relu"))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Визуализировать сетевую архитектуру
# SVG(model_to_dot(network, show_shapes=True).create(prog="dot",format="svg"))
# Сохранить визуализацию в виде файла
# plot_model(network, show_shapes=True, to_file="network.png")

# Классификация изображений
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# Сделать значение цветового канала первым
# K.set_image_data_format("channels_first")
# # Задать начальное значение для ГПСЧ
# np.random.seed(0)
# # Задать информацию об изображении
# channels = 1
# height = 28
# width = 28
# # Загрузить данные и цель из набора данных MNIST рукописных цифр
# (data_train, target_train), (data_test, target_test) = mnist.load_data()
# # Реформировать тренировочные данные об изображениях в признаки
# data_train = data_train.reshape(data_train.shape[0], channels, height, width)
# # Реформировать тестовые данные об изображениях в признаки
# data_test = data_test.reshape(data_test.shape[0], channels, height, width)
# # Прошкалировать пиксельную интенсивность в диапазон между 0 и 1
# features_train = data_train / 255
# features_test = data_test / 255
# # Преобразовать цель в кодировку с одним активным состоянием
# target_train = np_utils.to_categorical(target_train)
# target_test = np_utils.to_categorical(target_test)
# number_of_classes = target_test.shape[1]
# # Инициализировать нейронную сеть
# network = Sequential()
# # Добавить сверточный слой с 64 фильтрами, окном 5x5 и активационной функций ReLU
# network.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(channels, width, height), activation='relu'))
# # Добавить максимально редуцирующий слой с окном 2x2
# network.add(MaxPooling2D(pool_size=(2, 2)))
# # Добавить отсеивающий слой
# network.add(Dropout(0.5))
# # Добавить слой для сглаживания входа
# network.add(Flatten())
# # Добавить полносвязный слой из 128 блоков с активационной функцией ReLU
# network.add(Dense(128, activation="relu"))
# # Добавить отсеивающий слой
# network.add(Dropout(0.5))
# # Добавить полносвязный слой с активационной функцией softmax
# network.add(Dense(number_of_classes, activation="softmax"))
# # Скомпилировать нейронную сеть
# network.compile(loss="categorical_crossentropy", # Перекрестная энтропия
#                 optimizer="rmsprop", # Распространение CKO
#                 metrics=["accuracy"]) # Точностный показатель результативности
# # Натренировать нейронную сеть
# network.fit(features_train, # Признаки
#             target_train, # Цель
#             epochs=2, # Количество эпох
#             verbose=0, # Не печатать описание после каждой эпохи
#             batch_size=1000, # Количество наблюдений на пакет
#             validation_data=(features_test, target_test)) # Данные для оценивания

# Улучшение результативности с помощью расширения изображения(улучшить производительность сверточной нейронной сети)
# Для более оптимальных результатов предварительно обработать изображения и расширить(аргументировать|раздуть)
# данные с помощью класса-генератора ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
# Создать объект расширения изображения
# augmentation = ImageDataGenerator(featurewise_center=True,# Применить отбеливание ZCA
#                                   zoom_range=0.3,# Случайно приблизить изображения
#                                   width_shift_range=0.2,# Случайно сместить изображения
#                                   horizontal_flip=True,# Случайно перевернуть изображения
#                                   rotation_range=90)  # Случайно повернуть изображения
# Обработать все изображения из каталога 'raw/images'
# augment_images = augmentation.flow_from_directory("raw/images", # Папка с изображениями
#                                                   batch_size=32, # Размер пакета
#                                                   class_mode="binary", # Классы
#                                                   save_to_dir="processed/images")

# Классификация текста
# Использовать рекуррентную нейронную сеть с долгой краткосрочной памятью
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import models
from keras import layers
# Задать начальное значение для ГПСЧ
# np.random.seed(0)
# Задать желаемое количество признаков
# number_of_features = 1000
# Загрузить данные и вектор целей из данных с отзывами о кинофильмах
# (data_train, target_train), (data_test, target_test) = imdb.load_data(num_words=number_of_features)
# Использовать дополнение и усечение, чтобы каждое наблюдение имело 400 признаков
# features_train = sequence.pad_sequences(data_train, maxlen=400)
# features_test = sequence.pad_sequences(data_test, maxlen=400)
# Инициализировать нейронную сеть
# network = models.Sequential()
# Добавить встраивающий слой
# network.add(layers.Embedding(input_dim=number_of_features, output_dim=128))
# Добавить слой длинной краткосрочной памяти с 128 блоками
# network.add(layers.LSTM(units=128))
# Добавить полносвязный слой с сигмоидальной активационной функцией
# network.add(layers.Dense(units=1, activation="sigmoid"))
# Скомпилировать нейронную сеть
# network.compile(loss="binary_crossentropy", # Перекрестная энтропия
#                 optimizer="Adam", # Оптимизация Adam
#                 metrics=["accuracy"]) # Точностный показатель результативности
# Натренировать нейронную сеть
# history = network.fit(features_train, # Признаки
#                       target_train, # Цель
#                       epochs=3, # Количество эпох
#                       verbose=0, # Не печатать описание после каждой эпохи
#                       batch_size=1000, # Количество наблюдений на пакет
#                       validation_data=(features_test, target_test)) # Тестовые данные
