# ch_7_Работа с датами и временем

import numpy as np
import pandas as pd

# вектор строк(даты и время) преобразовать в данные временных рядов
# Создать строки
# date_strings = np.array(['03-04-2005 11:35 PM','23-05-2010 12:01 AM','04-09-2009 09:09 PM'])
# Конвертировать в метки datetime
# xc = [pd.to_datetime(date, format='%d-%m-%Y %I:%M %p') for date in date_strings]
# добавить аргумент в параметр errors для устранения проблем
# Конвертировать в метки datetime
# xc = [pd.to_datetime(date, format="%d-%m-%Y %I:%M %p", errors='coerce') for date in date_strings]
# Если errors="coerce" - любая проблема не вызовет ошибку, вместо этого установит значение, равным NaT

# добавить или изменить информацию о часовом поясе во временном ряду
# Создать метку datetime
# pd.Timestamp('2017-05-01 06:00:00', tz='Europe/London')
# Часовой пояс можно добавить к созданной метке вермени datetime(метод tz_localize)
# Создать метку datetime
# date = pd.Timestamp('2017-05-01 06:00:00')
# Задать часовой пояс
# date_in_london = date.tz_localize('Europe/London')
# Изменить часовой пояс
# date_in_london.tz_convert('Africa/Abidjan')

# одномерные объекты Series в pandas могут применять методы tz_iocaiize и tz_convert к каждому элементу
# Создать три даты
# dates = pd.Series(pd.date_range('2/2/2002', periods=3, freq='M'))
# Задать часовой пояс
# dates.dt.tz_localize('Africa/Abidjan')

# all_timezones - можно увидеть все строки, используемые для часовых поясов
# Загрузить библиотеку
from pytz import all_timezones
# Показать два часовых пояса
# xc = all_timezones[0:2]

# выбрать одну дату или несколько из вектора дат
# Создать фрейм данных
# dataframe = pd.DataFrame()
# Создать метки datetime
# dataframe['дата'] = pd.date_range('1/1/2001', periods=100000, freq='H')
# Выбрать наблюдения между двумя метками datetime
# xc = dataframe[(dataframe['дата'] > '2002-1-1 01:00:00') & (dataframe['дата'] <= '2002-1-1 04:00:00')]

# установить столбец даты как индекс фрейма данных, затем сделать срез с помощью метода loc
# Задать индекс
# dataframe = dataframe.set_index(dataframe['дата'])
# Выбрать наблюдения между двумя метками datetime
# dataframe.loc['2002-1-1 01:00:00':'2002-1-1 04:00:00']

# создать признаки для года, месяца, дня, часа и минуты из столбца дат и времени
# свойства времени объекта-получателя series.dt
# Создать фрейм данных
# dataframe = pd.DataFrame()
# Создать пять дат
# dataframe['дата'] = pd.date_range('1/1/2001', periods=150, freq='W')
# Создать признаки для года, месяца, дня, часа и минуты
# dataframe['год'] = dataframe['дата'].dt.year
# dataframe['месяц'] = dataframe['дата'].dt.month
# dataframe['день'] = dataframe['дата'].dt.day
# dataframe ['час'] = dataframe['дата'].dt.hour
# dataframe['минута'] = dataframe['дата'].dt.minute

# рассчитать время между двумя датами
# Создать фрейм данных
# dataframe = pd.DataFrame()
# Создать два признака datetime
# dataframe['Прибыло'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-04-2017')]
# dataframe['Осталось'] = [pd.Timestamp('01-01-2017'), pd.Timestamp('01-06-2017')]
# Вычислить продолжительность между признаками
# xc = dataframe['Осталось'] - dataframe['Прибыло']

# удалить из результата days и оставить только числовое значение
# Вычислить продолжительность между признаками
# pd.Series(delta.days for delta in (dataframe['Осталось'] - dataframe['Прибыло']))

# TimeDelta - это продолжительность между датами(между въездом и выездом)

# узнать день недели для каждой даты из вектора дат
# Создать даты
# dates = pd.Series(pd.date_range("2/2/2002", periods=3, freq="M"))
# Показать дни недели
# dates.dt.weekday_name
# числовой результат
# Показать дни недели
# dates.dt.weekday

# создать признак, который запаздывает на n периодов времени
# Создать фрейм данных
# dataframe = pd.DataFrame()
# Создать дату
# dataframe["даты"] = pd.date_range("1/1/2001", periods=5, freq="D")
# dataframe["цена акций"] = [1.1,2.2,3.3,4.4,5.5]
# Значения с запаздыванием на одну строку
# datafгаше["цена_акций_в_пре,шпущий_день" ] = dataframe["цена_акций"].shift(1)

# рассчитать некоторый статистический показатель для скользящего времени по временному ряду
# Создать метки datetim e
# time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# Создать фрейм данных, задать индекс
# dataframe = pd.DataFrame(index=time_index)
# Создать признак
# dataframe["цена_акций"] = [1,2,3,4,5]
# Вычислить скользящее среднее
# dataframe.rolling(window=2).mean()

# В данных временных рядов пропущены значения
# Создать дату
# time_index = pd.date_range("01/01/2010", periods=5, freq="M")
# Создать фрейм данных, задать индекс
# dataframe = pd.DataFrame(index=time_index)
# Создать признак с промежутком пропущенных значений
# dataframe["продажи"] = [1.0,2.0,np.nan,np.nan,5.0]
# Интерполировать пропущенные значения
# dataframe.interpolate()
# или
# заменить пропущенные значения последним перед промежутком известным значением(т. е. выполнить прямое заполнение)
# Прямое заполнение
# dataframe.ffill()
# или
# заменить пропущенные значения последним после промежутка известным значением(т.е. выполнить обратное заполнение)
# Обратное заполнение
# dataframe.bfill()