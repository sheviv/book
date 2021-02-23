# ch_8_Работа с изображениями


import cv2
import numpy as np
from matplotlib import pyplot as plt

# загрузить изображение для предобработки
from matplotlib import pyplot as plt
# Загрузить изображение в оттенках серого
# image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Показать изображение
# plt.imshow(image, cmap="gray")
# plt.axis("off")
# plt.show()

# Загрузить цветное изображение
# image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)

# OpenCV используется цветовая схема BGR, matplotlib — работают с RGB
# Загрузить цветное изображение
# image_bgr = cv2.imread("images/plane.jpg", cv2.IMREAD_COLOR)
# Конвертировать в RGB
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Показать изображение
# plt.imshow(image_rgb)

# сохранить изображение для предобработки
from matplotlib import pyplot as plt
# Загрузить изображение в оттенках серого
# image = cv2.imread("images/plane.jpg", cv2.IMREAD_GRAYSCALE)
# Сохранить изображение/ функция imwrite пишет поверх существующих файлов
# cv2.imwrite("images/plane_new.jpg", image)

# изменить размер изображения для дальнейшей предобработки
# resize для изменения размера изображения
# Загрузить изображение в оттенках серого
# image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Изменить размер изображения до 50 пикселов на 50 пикселов
# image_50x50 = cv2.resize(image, (50, 50))

# удалить внешнюю часть изображения, чтобы изменить его размеры
# отсечь кромки изображения путем нарезки массива
# Загрузить изображение в оттенках серого
# image = cv2.imread("images/plane_256x256.jpg", cv2.IMREAD_GRAYSCALE)
# Выбрать первую половину столбцов и все строки(ОБРЕЗАТЬ ПОЛОВИНУ ИЗОБРАЖЕНИЯ)
# image_cropped = image[:,:128]
# Показать изображение
# plt.imshow(image_cropped, cmap="gray")
# plt.axis("off")
# plt.show()

# сгладить изображение(размыть изображение)
# Загрузить изображение в оттенках серого
image = cv2.imread("Last.jpg", cv2.IMREAD_GRAYSCALE)
# Размыть изображение
# image_blurry = cv2.blur(image, (5,5))
# plt.imshow(image_blurry, cmap="gray")
# plt.axis("off")
# plt.show()

# подчеркнуть влияние размера ядра на степень размытости, ядро размытия 100х 100
# Размыть изображение
# image_very_blurry = cv2.blur(image, (100,100))
# Показать изображение
# plt.imshow(image_very_blurry, cmap="gray")
# plt.xticks([])
# plt.yticks([])
# plt.show()

# размывающее ядро/ Центральным элементом в ядре является исследуемый пиксел, а остальные — его соседи
# Создать ядро
# kernel = np.ones((5,5)) / 25.0
# Ядро можно применить к изображению вручную с помощью фильтра fiiter2D и создать аналогичный эффект размытия
# Применить ядро
# image_kernel = cv2.filter2D(image, -1, kernel)
# Показать изображение
# plt.imshow(image_kernel, cmap="gray")
# plt.xticks([])
# plt.yticks([])
# plt.show()

# увеличить резкость изображения
# Создать ядро, выделяющее целевой пиксел. Затем применить его к изображению с помощью фильтра fiiter2D
# Загрузить изображение в оттенках серого
# Создать ядро
# kernel = np.array([[0, -1, 0],[-1, 5,-1],[0 , - 1, 0]])
# Увеличить резкость изображения/ выделить сам пиксел.
# image_sharp = cv2.filter2D(image, -1, kernel)
# plt.imshow(image_sharp, cmap="gray")
# plt.axis("off")
# plt.show()

# усилить контрастность между пикселами изображения
# Выравнивание гистограммы — это инструмент обработки изображений(выделять объекты и фигуры)
# Загрузить изображение в оттенках серого
# Улучшить изображение
# image_enhanced = cv2.equalizeHist(image)
# Показать изображение
# plt.imshow(image_enhanced, cmap="gray")
# plt.axis("off")
# plt. show ()

# когда изображение — цветное, сначала преобразовать изображение в цветовой формат YUV: Y — это яркость, a U,V цвет
# Загрузить изображение
# image_bgr = cv2.imread("Last.jpg")
# Конвертировать в YUV
# image_yuv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YUV)
# Применить выравнивание гистограммы
# image_yuv[:, 0] = cv2.equalizeHist(image_yuv[:, 0])
# Конвертировать в RGB
# image_rgb = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
# Показать изображение
# plt.imshow(image_rgb)
# plt.axis("off")
# plt. show ()

# выделить в изображении цвет
# Определить диапазон цветов, а затем применить маску к изображению
# Загрузить изображение
# image_bgr = cv2.imread('Last.jpg')
# Конвертировать BGR в HSV
# image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
# Определить диапазон синих значений в HSV
# lower_blue = np.array([50,100,50])
# upper_blue = np.array([130,255,255])
# Создать маску
# mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
# Наложить маску на изображение
# image_bgr_masked = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)
# Конвертировать BGR в RGB
# image_rgb = cv2.cvtColor(image_bgr_masked, cv2.COLOR_BGR2RGB)
# Показать изображение
# plt.imshow(image_rgb)
# plt.axis("off")
# plt. show ()

# вывести упрощенную версию изображения
# Пороговая обработка(порогование) — установка пикселов с большей интенсивностью, чем некоторое значение,
# в белый цвет и меньшей — в черный цвет
# Загрузить изображение в оттенках серого
image_grey = cv2.imread("Last.jpg", cv2.IMREAD_GRAYSCALE)
# Применить адаптивную пороговую обработку
# max_output_value = 255
# neighborhood_size = 99
# subtract_from_mean = 10
# max_output_vaiue определяет максимальную интенсивность среди интенсивностей выходных пикселов.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C устанавливает порог пиксела как взвешенную сумму интенсивностей соседних пикселов
# neighborhood_size, subtract_from_mean - размер блока(размер окрестности, определения порога пиксела) и константа,
# вычитаемая из вычисленного порога(используется для ручной точной настройки порога).
# image_binarized = cv2.adaptiveThreshold(image_grey,
#                                         max_output_value,
#                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                         cv2.THRESH_BINARY,
#                                         neighborhood_size,
#                                         subtract_from_mean)
# Показать изображение
# plt.imshow(image_binarized, cmap="gray")
# plt.axis("off")
# plt.show()

# выделить на изображении передний план
# Отметить прямоугольник вокруг нужного переднего плана, затем выполнить алгоритм захвата и обрезки GrabCut
# Загрузить изображение и конвертировать в RGB
# image_bgr = cv2.imread('Last.jpg')
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Значения прямоугольника: начальная X, начальная Y, ширина, высота
# rectangle = (0, 56, 256, 150)
# Создать первоначальную маску
# mask = np.zeros(image_rgb.shape[:2], np.uint8)
# Создать временные массивы, используемые в grabCut
# bgdModel = np.zeros((1, 65), np.float64)
# fgdModel = np.zeros((1, 65), np.float64)
# Выполнить алгоритм grabCut
# cv2.grabCut(image_rgb,  # Наше изображение
#             mask,  # Маска
#             rectangle,  # Наш прямоугольник
#             bgdModel,  # Временный массив для фона
#             fgdModel,  # Временный массив для переднего плана
#             5,  # Количество итераций
#             cv2.GC_INIT_WITH_RECT)  # Инициализировать, используя прямоугольник
# Создать маску, где фоны уверенно или потенциально равны 0, иначе 1
# mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
# Умножить изображение на новую маску, чтобы вычесть фон
# image_rgb_nobg = image_rgb * mask_2[:, :, np.newaxis]
# Показать изображение
# plt.imshow(image_rgb_nobg)
# plt.axis("off")
# plt.show()

# найти края изображения
# детектор границ Джона Ф.Кэнни
# Загрузить изображение в оттенках серого
# image_gray = cv2.imread("Last.jpg", cv2.IMREAD_GRAYSCALE)
# Вычислить медиану интенсивности
# median_intensity = np.median(image_gray)
# Установить пороговые значения на одно стандартное отклонение
# выше и ниже медианы интенсивности
# lower_threshold = int(max(0, (1.0 - 0.33) * median_intensity))
# upper_threshold = int(min(255, (1.0 + 0.33) * median_intensity))
# Применить детектор границ Кэнни
# image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)
# Показать изображение
# plt.imshow(image_canny, cmap="gray")
# plt.axis("off")
# plt.show()

# обнаружить углы изображения(обнаружения пересечения двух краев)
# Использовать реализацию детектора углов Харриса
# Загрузить изображение в оттенках серого
# image_bgr = cv2.imread("Last.jpg")
# image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# image_gray = np.float32(image_gray)
# Задать параметры детектора углов
# block_size = 2
# aperture = 29
# free_parameter = 0.04
# Обнаружить углы
# detector_responses = cv2.cornerHarris(image_gray,
#                                       block_size,  # размер окрестности вокруг каждого пиксела, используемого для определения углов
#                                       aperture,  # размер используемого ядра Собела
#                                       free_parameter)  # более крупные значения соответствуют идентификации более "слабых" углов
# Крупные угловые маркеры
# detector_responses = cv2.dilate(detector_responses, None)
# Оставить только те отклики детектора, которые больше порога, пометить белым цветом
# threshold = 0.02
# image_bgr[detector_responses >
#           threshold *
#           detector_responses.max()] = [255, 255, 255]
# Конвертировать в оттенки серого
# image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
# Показать изображение
# plt.imshow(image_gray, cmap="gray")
# plt.axis("off")
# plt.show()

# преобразовать изображение в наблюдение для машинного самообучения
# метод flatten(NumPy) - преобразовать многомерный массив с данными изображения в вектор, содержащий значения наблюдения
# Загрузить изображение в оттенках серого
# images = cv2.imread("Last.jpg", cv2.IMREAD_GRAYSCALE)
# Изменить размер изображения до 10 пикселов на 10 пикселов
# image_10xl0 = cv2.resize(images, (10, 10))
# Конвертировать данные изображения в одномерный вектор
# image_10xl0.flatten()
# plt.imshow(image_10xl0, cmap="gray")
# plt.axis("off")
# plt. show ()

# признак, основанный на цветах изображения(получим 3значения признаков, для каждого цветового канала изображения)
# пиксел изображения - комбинация нескольких цветовых каналов(три:к,з,с). Вычислить средние значения к,з,c каналов
# для изображения - создать три цветовых признака(средние цвета в этом изображении)
# Загрузить изображение как BGR
# image_bgr = cv2.imread("Last.jpg", cv2.IMREAD_COLOR)
# Вычислить среднее значение каждого канала
# channels = cv2.mean(image_bgr)
# Поменять местами синее и красное значения (переведя в RGB, а не BGR)
# observation = np.array([(channels[2], channels [1], channels[0])])
# Показать изображение
# plt.imshow(observation)
# plt.axis("off")
# plt.show()

# создать набор признаков, представляющих цвета на изображении
# Вычислить гистограммы для каждого цветового канала
# Загрузить изображение
# image_bgr = cv2.imread("Last.jpg", cv2.IMREAD_COLOR)
# Конвертировать в RGB
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
# Создать список для значений признаков
# features = []
# Вычислить гистограмму для каждого цветового канала
# colors = ("r", "g", "b")
# Для каждого цветового канала:
# вычислить гистограмму и добавить в список значений признаков
# for i, channel in enumerate(colors):
#     histogram = cv2.calcHist([image_rgb],  # Изображение
#                              [i],  # Индекс канала
#                              None,  # Маска отсутствует
#                              [256],  # Размер гистограммы
#                              [0, 256])  # Диапазон
#     features.extend(histogram)
# Создать вектор для значений признаков наблюдения
# observation = np.array(features).flatten()

# х - 256 возможных значений канала, у — количество появлений определенного значения канала во всех пикселах изображения
# colors = ("r", "g", "b")
# Для каждого цветового канала:
# вычислить гистограмму и добавить в список значений признаков
# for i, channel in enumerate(colors):
#     histogram = cv2.calcHist([image_rgb],  # Изображение
#                              [i],  # Индекс канала
#                              None,  # Маска отсутствует
#                              [256],  # Размер гистограммы
#                              [0, 256])  # Диапазон
#     plt.plot(histogram, color=channel)
#     plt.xlim([0, 256])
# Показать график
# plt.show()