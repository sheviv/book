# Классификация отзывов к фильмам: пример бинарной классификации

import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers

# в обучающих данных будет сохранено 10000 слов
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# word_index — словарь, отображающий слова в целочисленные индексы
world_index = imdb.get_word_index()
# обратное представление словаря, отображающее индексы в слова
reverse_word_index = dict([(value, key) for (value, key) in world_index.items()])
# Декодирование отзыва, индексы смещены на 3, т.к. индексы 0, 1 и 2 зарезервированы слов:
# «padding» (отступ), «start of sequence» (начало последовательности) и «unknown» (неизвестно)
decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])


# Подготовка данных
# Кодирование последовательностей целых чисел в бинарную матрицу
# Создание матрицы с формой (len(sequences),dimension)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Векторизованные обучающие данные
x_train = vectorize_sequences(train_data)
# Векторизованные контрольные данные
x_test = vectorize_sequences(test_data)
# векторизовать метки
y_train = np.array(train_labels).astype('float32')
y_test = np.array(test_labels).astype('float32')

# Конструирование сети
# Определение модели
# model = models.Sequential()
# model.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# Компиляция модели
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

# Настройка оптимизатора
# from keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# Использование нестандартных функций потерь и метрик
# from keras import losses
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss=losses.binary_crossentropy,
#               metrics=[metrics.binary_accuracy])

# Проверка решения
# Создание проверочного набора - валидационный набор
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Обучение модели
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['acc'])
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))
# history_dict = history.history
# history_dict.kesy()  # четыре элемента — по одному на метрику

# Формирование графиков потерь на этапах обучения и проверки
import matplotlib.pyplot as plt
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, 20 + 1)
# plt.plot(epochs, loss_values, 'bo', label='Training loss')  # «blue dot» — «синяя точка»
# plt.plot(epochs, val_loss_values, 'b', label='Validation loss')  # «solid blue line» — «сплошная синяя линия»
# plt.title('Training and val loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# Формирование графиков точности на этапах обучения и проверки
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']
# plt.plot(epochs, acc_values, 'bo', label='Training acc')
# plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# Обучение новой модели с нуля без переобучения с 4 эпохами
models = models.Sequential()
models.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
models.add(layers.Dense(16, activation='relu'))
models.add(layers.Dense(1, activation='sigmoid'))

models.compile(optimizer='rmsprop',
               loss='binary_crossentropy',
               metrics=['accuracy'])
models.fit(x_train, y_train, epochs=4, batch_size=512)
results = models.evaluate(x_test, y_test)

# Использование обученной сети для предсказаний на новых данных
print(models.predict(x_test))