# Заглянем глубже: наивная байесовская классификация

# Наивные байесовские модели — группа исключительно быстрых и простых алгоритмов классификации,
# зачастую подходящих для наборов данных очень высоких размерностей.

# Байесовская классификация
# порождающей моделью(generative model) - определяет гипотетический случайный процесс генерации данных
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Гауссов наивный байесовский классификатор(данные всех категорий взяты из простого нормального распределения)
from sklearn.datasets import make_blobs
# X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
# plt.show()
# для каждой из меток с ростом вероятности по мере приближении к центру эллипса
from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(X, y)
# Сгенерируем новые данные и выполним предсказание метки
# rng = np.random.RandomState(0)
# Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
# ynew = model.predict(Xnew)
# график этих новых данных и понять, где пролегает граница принятия решений(decision boundary)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')
# lim = plt.axis()
# plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu',alpha=0.1)
# plt.axis(lim)
# plt.show()  # граница изогнута, при Гауссовом наивном байесовском классификаторе соответствует кривой второго порядка
# возможности естественной вероятностной классификации
# yprob = model.predict_proba(Xnew)
# yprob[-8:].round(2)  # Столбцы отражают апостериорные вероятности первой и второй меток соответственно

# Полиномиальный наивный байесовский классификатор
# from sklearn.datasets import fetch_20newsgroups
# data = fetch_20newsgroups()
# data_t = data.target_names
# print(data_t)
# categories = ['talk.religion.misc', 'soc.religion.christian', 'sci.space', 'comp.graphics']
# train = fetch_20newsgroups(subset='train', categories=categories)
# test = fetch_20newsgroups(subset='test', categories=categories)
# создадим конвейер, присоединяющий его последовательно к полиномиальному наивному байесовскому классификатору
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# применить модель к обучающей последовательности и предсказать метки для контрольных данных
# model.fit(train.data, train.target)
# labels = model.predict(test.data)
# эффективность работы оценивателя - матрица различий между настоящими и предсказанными метками для контрольных данных
# from sklearn.metrics import confusion_matrix
# mat = confusion_matrix(test.target, labels)
# sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=train.target_names, yticklabels=train.target_names)
# plt.xlabel('true label')
# plt.ylabel('predicted label')
# plt.show()
# фрагмент кода описывает простую вспомогательную функцию, возвращающую предсказание для отдельной строки
# def predict_category(s, train=train, model=model):
#     pred = model.predict([s])
#     return train.target_names[pred[0]]
# predict_category('sending a payload to the ISS')  # example

# наивный байесовский классификатор
#     - они выполняют как обучение, так и предсказание исключительно быстро;
#     - обеспечивают простое вероятностное предсказание;
#     - их результаты часто очень легки для интерпретации;
#     - у них очень мало (если вообще есть) настраиваемых параметров.
# хорошие результаты в следующих случаях
#     - когда данные действительно соответствуют наивным допущениям (на практике бывает очень редко)
#     - для очень хорошо разделяемых категорий, когда сложность модели не стольважна
#     - для данных с очень большим числом измерений, когда сложность модели не столь важна
