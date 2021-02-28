# ядерная оценка плотности распределения(KDE)

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Обоснование метода KDE: гистограммы
# def make_data(N, f=0.3, rseed=1):
#     rand = np.random.RandomState(rseed)
#     x = rand.randn(N)
#     x[int(f * N):] += 5
#     return x
# x = make_data(1000)
# hist = plt.hist(x, bins=30, density=True)  # density - высота интервалов не число точек,а плотность вероятности
# plt.show()  # Данные, полученные из сочетания нормальных распределений