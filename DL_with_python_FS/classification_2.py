# Классификация новостных лент: пример классификации в несколько классов

from keras.datasets import reuters
from keras import models
from keras import layers
import numpy as np

# Загрузка данных Reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# Декодирование новостей обратно в текст
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (value, key) in word_index.items()])
# индексы смещены на 3, т.к. индексы 0, 1 и 2 зарезервированы слов:
# «padding» (отступ), «start of sequence» (начало последовательности) и «unknown» (неизвестно)
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# Кодирование данных
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


x_train = vectorize_sequences(train_data)  # Векторизованные обучающие данные
x_test = vectorize_sequences(test_data)  # Векторизованные контрольные данные
# кодирование меток конструирование вектора с нулевыми элементами со значением 1 в элементе,
# индекс которого соответствует индексу метки.
# def to_one_hot(labels, dimension=46):
#     results = np.zeros((len(labels), dimension))
#     for i, label in enumerate(labels):
#         results[i, label] = 1.
#     return results
# one_hot_train_labels = to_one_hot(train_labels)
# one_hot_test_labels = to_one_hot(test_labels)
# реализация в Keras
from keras.utils.np_utils import to_categorical
one_hot_train_labels = to_categorical(train_labels)  # Векторизованные обучающие данные
one_hot_test_labels = to_categorical(test_labels)  # Векторизованные контрольные данные

# Конструирование сети
# Определение модели
# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(46, activation='softmax'))

# Компиляция модели
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# Создание проверочного набора
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

# Обучение модели
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=20,
#                     batch_size=512,
#                     validation_data=(x_val, y_val))

# Формирование графиков потерь на этапах обучения и проверки
import matplotlib.pyplot as plt
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# Формирование графиков точности на этапах обучения и проверки
# history_dict = history.history
# acc = history_dict['acc']
# val_acc = history.history['val_acc']
# plt.plot(20, acc, 'bo', label='Training acc')
# plt.plot(20, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# Обучение новой модели с нуля
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(partial_x_train,
          partial_y_train,
          epochs=9,
          batch_size=512,
          validation_data=(x_val, y_val))
result = model.evaluate(x_test, one_hot_test_labels)

# Предсказания на новых данных
# Получение предсказаний для новых данных
predictions = model.predict(x_test)
# Наибольший элемент, элемент с наибольшей вероятностью, — это предсказанный класс:
np.argmax(predictions[0])


# Другой способ обработки меток и потерь
# метки можно преобразовать в тензор целых чисел
x_train = np.array(train_labels)
y_test = np.array(test_labels)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])


# Модель с узким местом для информации
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))

# Метки в многоклассовой классификации можно обрабатывать двумя способами:
# 1 - Кодировать метки с применением метода кодирования категорий (также известного как прямое кодирование) и
# использовать функцию потерь categorical_crossentropy.
# 2 - Кодировать метки как целые числа и использовать функцию потерь sparse_categorical_crossentropy.

