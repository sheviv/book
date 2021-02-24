# ch_2_Загруска данных

import pandas as pd
from sqlalchemy import create_engine

# импортировать файл значений с разделителями-запятыми
# Создать URL-адрес
# url = 'https://tinyurl.com/simulated-data'
# Загрузить набор данных
# dataframe = pd.read_csv(url)
# Взглянуть на первые две строки
# print(dataframe.head(2))

# загрузить электронную таблицу Excel
# Создать URL-адрес
# url = 'https://tinyurl.com/simulated-excel'
# Загрузить данные
# dataframe = pd.read_excel(url, sheet_name=0, header=l) # sheet name - указывает,на лист в файле Excel
# Взглянуть на первые две строки
# dataframe.head(2)

# загрузить файл JSON для предобработки данных
# Создать URL-адрес
# url = 'https://tinyurl.com/simulated-json'
# Загрузить данные
# dataframe = pd.readjson(url, orient=' columns')
# orient - указывает pandas, как структурирован файл JSON
# json normaiize - преобраовывает полуструктурированные данные JSON во фрейм данных pandas

# загрузить данные из базы данных с помощью языка структурированных запросов (SQL)
# Создать подключение к базе данных
# database_connection = create_engine('sqlite:///sample.db') /create_engine - подключения к ядру базы данных SQL(SQLite)
# Загрузить данные
# dataframe = pd.read_sql_query('SELECT * FROM data', database_connection)
# read_sqi_query - опрос базы данных с помощью SQL и поместить результаты во фрейм данных
# 'SELECT * FROM data - просит базу данных предоставить нам все столбцы (*) из таблицы data
# Взглянуть на первые две строки
# data frame.head(2)
