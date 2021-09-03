# ch_9_Снижение размерности с помощью выделения признаков

# сократить количество признаков, сохраняя при этом дисперсию данных
# Использовать анализ главных компонент с помощью класса р с а библиотеки scikit-leam
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
# Загрузить данные
# digits = datasets.load_digits()
# Стандартизировать матрицу признаков
# features = StandardScaler().fit_transform (digits.data)
# Создать объект PCA, который сохранит 99% дисперсии
# pca = PCA(n_components=0.99, whiten=True)
# Выполнить анализ РСА
# features_pca = pca.fit_transform(features)
# Показать результаты
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_pca.shape[1])

# данные линейно неразделимы, и требуется сократить размерности
# Применить расширение анализа главных компонент, в котором используются ядра для нелинейного уменьшения размерности
# Загрузить библиотеки
# from sklearn.decomposition import PCA, KernelPCA
# from sklearn.datasets import make_circles
# Создать линейно неразделимые данные
# features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# Применить ядерный PCA с радиально-базисным функциональным ядром (RBF-ядром)
# kpca = KernelPCA(kernel="rbf", gaitma=15, n_components=1)
# features_kpca = kpca.fit_transform(features)
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_kpca.shape[1])

# сократить признаки, используемые классификатором
# линейный дискриминантный анализ(linear discriminant analysis, LDA) - спроецировать объекты на оси компонент,
# которые максимизируют разделение классов
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Загрузить набор данных цветков ириса:
# iris = datasets.load_iris()
# features = iris.data
# target = iris.target
# Создать объект и выполнить LDA, затем использовать его для преобразования признаков
# lda = LinearDiscriminantAnalysis(n_components=1)
# features_lda = lda.fit(features, target).transform(features)
# Напечатать количество признаков
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_lda.shape[1])
# просмотра объема дисперсии
# xc = lda.explained_variance_ratio_

# уменьшить размерность матрицы признаков с неотрицательными значениями
# разложение неотрицательной матрицы(non-negative matrix factorization, NMF) - уменьшая размерности матрицы признаков
# Загрузить библиотеки
from sklearn.decomposition import NMF
from sklearn import datasets
# Загрузить данные
# digits = datasets.load_digits()
# Загрузить матрицу признаков
# features = digits.data
# Создать NMF и выполнить его подгонку
# nmf = NMF(n_components=10, random_state=1)
# features_nmf = nmf.fit_transform(features)
# Показать результаты
# print("Исходное количество признаков:", features.shape[1])
# print("Сокращенное количество признаков:", features_nmf.shape[1])

# уменьшить размерность разреженной матрицы признаков
# усеченное сингулярное разложение (truncated singular value decomposition, TSVD)
# Загрузить библиотеки
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np
# Загрузить данные
# digits = datasets.load_digits()
# Стандартизировать матрицу признаков
# features = StandardScaler().fit_transform(digits.data)
# Сделать разреженную матрицу
# features_sparse = csr_matrix(features)
# Создать объект TSVD
# tsvd = TruncatedSVD(n_components=10)
# Выполнить TSVD на разреженной матрице
# features_sparse_tsvd = tsvd.fit(features_sparse).transform(features_sparse)
# Показать результаты
# print("Исходное количество признаков:", features_sparse.shape[1])
# print("Сокращенное количество признаков:", features_sparse_tsvd.shape[1])

# функцию, которая выполняет усеченное сингулярное разложение с компонентами n_components,
# установленными на единицу меньше, чем количество исходных признаков, затем вычислить количество компонент,
# которые объясняют желаемый объем дисперсии исходных данных
# Загрузить данные
# digits = datasets.load_digits()
# Стандартизировать матрицу признаков
# features = StandardScaler().fit_transform(digits.data)
# Сделать разреженную матрицу
# features_sparse = csr_matrix(features)
# Создать и выполнить TSVD с числом признаков меньше на единицу
# tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
# features_tsvd = tsvd.fit(features)
# Поместить в список объясненные дисперсии
# tsvd_var_ratios = tsvd.explained_variance_ratio_
# Создать функцию
# def select_n_components(var_ratio, goal_var):
    # Задать исходную объясненную на данный момент дисперсию
    # total_variance = 0.0
    # Задать исходное количество признаков
    # n_components = 0
    # Для объясненной дисперсии каждого признака:
    # for explained_variance in var_ratio:
        # Добавить объясненную дисперсию к итогу
        # total_variance += explained_variance
        # Добавить единицу к количеству компонент
        # n_components += 1
        # Если достигнут целевой уровень объясненной дисперсии
        # if total_variance >= goal_var:
            # Завершить цикл
            # break
    # Вернуть количество компонент
    # return n_components
# Выполнить функцию
# select_n_components(tsvd_var_ratios, 0.95)

