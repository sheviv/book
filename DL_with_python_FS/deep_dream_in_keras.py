# Реализация DeepDream в Keras
# DeepDream — это метод художественной обработки изображений

# Загрузка предварительно обученной модели Inception V3
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
import scipy
from keras.preprocessing import image

K.set_learning_phase(0)  # не обучать модель - запретить все операции имеющие отношение к обучению
# Конструирование сети Inception V3 без сверточной основы. Модель будет загружаться с весами,
# полученными в результате предварительного обучения на наборе ImageNet
model = InceptionV3(weights='imagenet', include_top=False)
# Настройка конфигурации DeepDream
layer_contributions = {'mixed2': 0.2, 'mixed3': 3., 'mixed4': 2., 'mixed5': 1.5}
# Словарь - имена слоев в коэффициенты, определяющие вклады слоев в потери,для максимизации.
# имена слоев жестко «зашиты» во встроенное приложение Inception V3(список имен всех слоев - model.summary())
# Определение потерь для максимизации
layer_dict = dict([(layer.name, layer) for layer in model.layers])  # словарь, отображающий имена слоев в их экземпляры
loss = K.variable(0.)  # Величина потерь определяется добавлением вклада слоя в эту скалярную переменную
for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output  # Получение результата слоя
    # Добавление L2 нормы признаков слоя к потерям. избежать влияния рамок,
    # в подсчете потерь участвуют только пикселы, не попадающие на рамку
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling
    loss.assign_add(coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling)
# Процесс градиентного восхождения
dream = model.input  # тензор хранит сгенерированное изображение
grads = K.gradients(loss, dream)[0]  # Вычисление градиентов изображения с учетом потерь
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)  # Нормализация градиентов
# Настройка функции Keras для извлечения значения потерь и градиентов для заданного исходного изображения
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)
def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values
# выполняет заданное число итераций градиентного восхождения
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

# функция, открывающая изображение, изменяющая его размер и преобразующая в тензор для обработки в Inception V3
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = InceptionV3.preprocess_input(img)
    return img
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,float(size[0]) / img.shape[1],float(size[1]) / img.shape[2],1)
    return scipy.ndimage.zoom(img, factors, order=1)
# функция, преобразующая тензор в допустимое изображение
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        # Отмена предварительной обработки, выполненной вызовом inception_ v3.preprocess_input
        x = x.reshape((x.shape[1], x.shape[2], 3))
        x /= 2.
        x += 0.5
        x *= 255.
        x = np.clip(x, 0, 255).astype('uint8')
    return x
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


# Выполнение градиентного восхождения через последовательность масштабов
import numpy as np
# гиперпараметры
step = 0.01  # Размер шага градиентного восхождения
num_octave = 3  # Количество масштабов, на которых выполняется градиентное восхождение
octave_scale = 1.4  # Отношение между соседними масштабами
iterations = 20  # Число шагов восхождения для каждого масштаба
max_loss = 10.  # величина потерь > 10 - прервать процесс градиентного восхождения, чтобы избежать появления артефактов
base_image_path = '/DL_with_python_FS/123.png'  # путь к файлу изображения
img = preprocess_image(base_image_path)  # Загрузка базового изображения в массив Numpy
original_shape = img.shape[1:3]
# Подготовка списка кортежей shape, определяющих разные масштабы для выполнения градиентного восхождения
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i))
        for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]  # Переворачивание списка кортежей shape, чтобы следовали в порядке ув. масштаба
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])  # Уменьшение изображения в массиве Numpy до мин. масштаба
for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)  # Увеличение масштаба изображения
    # Выполнение градиентного восхождения с изменением изображения
    img = gradient_ascent(img,iterations=iterations,step=step,max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    # Вычисление высококачественной версии исходного изображения с заданными размерами
    same_size_original = resize_img(original_img, shape)
    # Разница между двумя изображениями — это детали, утерянные в результате масштабирования
    lost_detail = same_size_original - upscaled_shrunk_original_img
    img += lost_detail  # Повторное внедрение деталей
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='dream_at_scale_' + str(shape) + '.png')
save_img(img, fname='final_dream.png')
