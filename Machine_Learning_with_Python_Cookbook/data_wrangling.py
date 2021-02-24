# ch_3_Упорядочение данных

import pandas as pd
import numpy as np

# создать новый фрейм данных
# Создать фрейм данных DataFrame
# dataframe = pd.DataFrame()
# Добавить столбцы
# dataframe['Имя'] = ['Джеки Джексон', 'Стивен Стивенсон']
# dataframe['Возраст'] = [38, 25]
# dataframe['Водитель'] = [True, False]

# добавить новые строки в конец фрейма данных
# Создать строку
# new_person = pd.Series(['Молли Муни', 40, True], index=['Им я ','Возраст','Водитель'])
# Добавить строку в конец фрейма данных
# dataframe.append(new_person, ignore_index=True)

# описательня статистику для любых числовых столбцов
# Показать статистику
# dataframe.describe()

# выбор одной или нескольких строк либо значений использовать методы 1ос или iloc
# Выбрать первую строку
# dataframe.iloc[0]
# Выбрать три строки
# dataframe.iloc[1:4]
# все строки до определенной точки(до 4 строчки)
# dataframe.iloc[:4]

# задать индекс на именах пассажиров
# dataframe = dataframe.set_index(dataframe['Name'])
# Показать строку
# dataframe.loc['Allen, Miss Elisabeth Walton']

# 1ос - индекс фрейма является меткой(например, строковым значением)
# iloc - отыскивает позицию во фрейме данных(например, iloc[0] вернет первую строку)

# отобрать строки фрейма данных на основе некоторого условия
# Показать верхние две строки, где столбец 'sex' равняется 'female'
# dataframe[dataframe['Sex'] == 'female'].head(2)
# множественное условие(пример: женщины 65 лет и старше)
# dataframe[(dataframe['Sex'] == 'female') & (dataframe['Age'] >= 65)]

# замена значений во фрейме данных
# заменить значения "female" в столбце "sex" на "woman"
# dataframe['Sex'].replace("female", "Woman").head(2)

# замена по всему объекту DataFrame
# dataframe.replace(1, "One").head(2) || dataframe.replace(r"lst", "First", regex=True).head(2)

# переименовать столбец во фрейме данных pandas
# dataframe.rename(columns={'PClass': 'Passenger Class'}).head(2)
# для нескольких
# dataframe.rename (columns={'PClass': 'Passenger Class', 'Sex': 'Gender'}).head(2)

# минимум, максимум, сумму, среднее арифметическое и количество значений числового столбца
# print('Максимум:', dataframe['Age'].шах())
# print('Минимум:', dataframe['Age'].min())
# print('Среднее:', dataframe['Age'].mean())
# print('Сумма:', dataframe['Age'].sum())
# print('Количество:', dataframe['Age'].count())

# Показать количества значений по всему фрейму
# dataframe.count()

# выбрать все уникальные значения в столбце
# Выбрать уникальные значения
# dataframe['Sex'].unique()

# покать все уникальные значения вместе с количеством появлений каждого значения
# Показать количества появлений
# dataframe['Sex'].value_counts()

# выбрать пропущенные значения во фрейме данных
# dataframe[dataframe['Age'].isnull()].head(2)

# Заменить значения с NaN
# dataframe['Sex'] = dataframe['Sex'].replace('male', np.nan)

# Загрузить данные, задать пропущенные значения
# dataframe = pd.read_csv(url, na_values=[np.nan, 'NONE', -999])

# удалить столбец из фрейма данных
# применить метод drop с параметром axis=1
# Удалить столбец
# dataframe.drop('Age', axis=1).head(2)

# Отбросить столбцы
# dataframe.drop(['Age', 'Sex'], axis=1)

# отбросить столбец по индексу с помощью массива dataframe.columns
# dataframe.drop(dataframe.columns[1], axis=1)

# удалить одну или несколько строк из фрейма данных
# Удалить строки, показать первые две строки вывода
# dataframe[dataframe['Sex'] != 'male'].head(2)

# Удалить строк по имени || по ее индексу
# dataframe[dataframe['Name'] != 'Allison, Miss Helen Loraine'].head(2) || dataframe[dataframe.index != 0].head(2)

# удалить повторяющиеся строки из фрейма данных
# Удалить дубликаты, показать первые две строки вывода
# dataframe.drop_duplicates().head(2)
# drop_duplicates - отбрасывает только те строки, которые идеально совпадают по всем столбцам

# Удалить дубликаты по столбцу
# dataframe.drop_duplicates(subset=['Sex'])

# Сгруппировать строки по значениям столбца 'Sex', вычислить среднее каждой группы
# dataframe.groupby('Sex').mean()

# Сгруппировать строки, вычислить среднее
# dataframe.groupby(['Sex', 'Survived'])['Age'].mean()

# сгруппировать отдельные строки по периодам времени
# Создать диапазон дат
# time_index = pd.date_range('06/06/2017', periods=100000, freq='30S')
# Создать фрейм данных
# dataframe = pd.DataFrame(index=time_index)
# Создать столбец случайных значений
# dataframe['Sale_Amount'] = np.random.randint(1, 10, 100000)
# Сгруппировать строки по неделе, вычислить суьму за неделю
# dataframe.resample('W').sum()

# Сгруппировать по двум неделям, вычислить среднее
# dataframe.resample('2W').mean()

# Сгруппировать по месяцу, подсчитать строки
# dataframe.resample('М').count()

# итерация по каждому элементу в столбце и применить какое-то действие
# Напечатать первые два имени в верхнем регистре
# for name in dataframe['Name'][0:2]:
#     print(name.upper())
# Напечатать первые два имени в верхнем регистре
# [name.upper() for name in dataframe!'Name'][0:2]]

# применить некую функцию ко всем элементам в столбце
# Создать функцию
# def uppercase(x):
#     return x.upper()
# Применить функцию, показать две строки
# dataframe['Name'].apply(uppercase)[0:2]

# применить функцию к каждой группе/ apply - применить функцию к группам
# Объединить методы groupby и apply
# Сгруппировать строки, применить функцию к группам
# dataframe.groupby('Sex').apply(lambda x: x.count())

# связать последовательно два фрейма данных
# конкатенации фреймов вдоль оси строк использовать метод concat с параметром axis=o
# Конкатенировать фреймы данных по строкам
# pd.concat([dataframe_a, dataframe_b], axis=0)
# Для конкатенации вдоль оси столбцов можно использовать axis=i
# Конкатенировать фреймы данных по столбцам
# pd.concat([dataframe_a, dataframe_b], axis=1)

# добавление новой строки во фрейм данных
# Создать строку
# row = pd.Series([10, 'Chris', 'Chillon'], index=['id', 'first', 'last'])
# Добавить строку в конец
# dataframe_a.append(row, ignore_index=True)

# слияние/единение/склеивание двух фреймов данных(внутренне соединение)
# Создать фрейм данных
# employee_data = {'employee_id': ['1'],'name': ['Amy Jones']}
# dataframe_employees = pd.DataFrame(employee_data, columns=['employee_id', 'name'])
# Создать фрейм данных
# sales_data = {'employee_id': ['3'], 'total_sales': [23456]}
# dataframe_sales = pd.DataFrame(sales_data, columns=['employee_id'])
# Выполнить слияние фреймов данных
# pd.merge(dataframe_employees, dataframe_sales, on='employee_id')

# слияние/единение/склеивание двух фреймов данных(внешнее соединение)
# Выполнить слияние фреймов данных
# pd.merge(dataframe_employees, dataframe_sales, on='employee_id', how='outer')