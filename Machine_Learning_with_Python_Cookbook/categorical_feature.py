# ch_5_Работа с категориальными данными

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer

# признак с номинальными классами, который не имеет внутренней упорядоченности(например, яблоко, груша, банан)
# Преобразовать признак в кодировку с одним активным состоянием с помощью класса LabelBinarizer библиотеки scikit-lean
# Создать признак
xc = np.array([["Texas"],["California"],["Texas"],["Delaware"],["Texas"]])
# Создать кодировщик одного активного состояния
# one_hot = LabelBinarizer()
# Преобразовать признак в кодировку с одним активным состоянием
# one_hot.fit_transform(xc)

# обратить кодирование с одним активным состоянием
# one_hot.inverse_transform(one_hot.transform(xc))

# преобразование признака в кодировку с одним активным состоянием
# Создать фиктивные переменные из признака
# pd.get_dummies(xc[:,0])

# преобразование признака в кодировку с несколькими активными состояниями
# Создать мультиклассовый признак
multiclass_feature = [("Texas", "Florida"),("California", "Alabama"),("Texas", "Florida"),("Delware", "Florida"),("Texas", "Alabama")]
# Создать мультиклассовый кодировщик, преобразующий признак в кодировку с одним активным состоянием
# one_hot_multiclass = MultiLabelBinarizer()
# Кодировать мультиклассовый признак в кодировку с одним активным состоянием
# one_hot_multiclass.fit_transform(multiclass_feature)
# Взглянуть на классы
# one_hot_multiclass.classes_

# Использовать метод replace фрейма данных pandas для преобразования строковых меток в числовые эквиваленты:
# Создать признаки
dataframe = pd.DataFrame({"оценка": ["низкая", "низкая", "средняя", "средняя", "высокая"]})
# Создать словарь преобразования шкалы
# scale_mapper = {"низкая": 1, "средняя": 2, "высокая": 3}
# Заменить значения признаков значениями словаря
# dataframe["оценка"].replace(scale_mapper)

# конвертировать словарь в матрицу признаков
# Импортировать библиотеку
from sklearn.feature_extraction import DictVectorizer
# Создать словарь
data_dict = [{"красный": 2, "синий": 4},{"красный": 4, "синий": 3},{"красный": 1, "желтый": 2},{"красный": 2, "желтый": 2}]
# Создать векторизатор словаря
# dictvectorizer = DictVectorizer(sparse=False)
# Конвертировать словарь в матрицу признаков
# features = dictvectorizer.fit_transform(data_dict)
# Получить имена признаков
# feature_names = dictvectorizer.get_feature_names()
# Создать фрейм данных из признаков
# pd.DataFrame(features, columns=feature_names)

# создать матрицу признаков, где каждый признак — это количество вхождений слова в каждый документ
# Создать словари частотностей слов для четырех документов
# doc_l_word_count = {"красный": 2, "синий": 4}
# doc_2_word_count = {"красный": 4, "синий": 3}
# doc_3_word_count = {"красный": 1, "желтый": 2}
# doc_4_word_count = {"красный": 2, "желтый": 2}
# Создать список
# doc_word_counts = [doc_l_word_count,doc_2_word_count,doc_3_word_count,doc_4_word_count]
# Конвертировать список словарей частотностей слов в матрицу признаков
# dictvectorizer.fit_transform(doc_word_counts)

# категориальный признак, содержащий пропущенные значения, которые требуется заменить предсказанными значениями
from sklearn.neighbors import KNeighborsClassifier
# Создать матрицу признаков с категориальным признаком
# X = np.array([[0, 2.10, 1.45],[1, 1.18, 1.33],[0, 1.22, 1.27],[1, -0.21, -1.19]])
# Создать матрицу признаков с отсутствующими значениями в категориальном признаке
# X_with_nan = np.array([[np.nan, 0.87, 1.31],[np.nan, -0.67, -0.22]])
# Натренировать ученика KNN
# elf = KNeighborsClassifier(3, weights='distance')
# trained_model = elf.fit(X[:,1:], X[:,0])
# Предсказать класс пропущенных значений
# imputed_values = trained_model.predict(X_with_nan[1:])
# Соединить столбец предсказанного класса с другими признаками
# X_with_imputed = np.hstack((imputed_values.reshape(-1,1), X_with_nan[:,1:]))
# Соединить две матрицы признаков
# np.vstack((X_with_imputed, X))
# или
# заполнение пропущенных значений наиболее частыми значениями признаков
from sklearn.impute import SimpleImputer
# Соединить две матрицы признаков
# X_complete = np.vstack((X_with_nan, X))
# imputer = SimpleImputer(strategy='most_frequest', axis=0)
# imputer.fit_transform(X_complete)