# ch_6_Работа с текстом

# элементарная очистка неструктурированных текстовых данных
# Создать текст
# text_data = [" Interrobang. By Aishwarya Henriette ","Parking And Going. By Karl Gautier"," Today Is The night. By Jarek Prakash "]
# Удалить пробелы
# strip_whitespace = [string.strip() for string in text_data]
# Удалить точки
# remove_periods = [string.replace(".", "") for string in strip_whitespace]
# функция преобразования
# Создать функцию
# def capitalizer(string: str) -> str:
#     return string.upper()
# Применить функцию
# [capitalizer(string) for string in remove_periods]
# регулярные выражения
# Импортировать библиотеку
import re
# Создать функцию
# def replace_letters_with_X(string: str) -> str:
#     return re.sub(r"[a-zA-Z]", "X", string)
# Применить функцию
# [replace_letters_with_X(string) for string in remove_periods]

# извлечь только текст из текстовых данных с элементами HTML
# набор функциональных средств библиотеки Beautiful Soup для анализа и извлечения текста из HTML
# Загрузить библиотеку
from bs4 import BeautifulSoup
# Создать немного кода с разметкой HTML
# html = """
# <div class='full_name'><span style='font-weight:bold'>
# Masego</span> Azra</div>"
# """
# Выполнить разбор html
# soup = BeautifulSoup(html, "lxml")
# Найти элемент div с классом "full_name", показать текст
# soup.find("div", {"class": "full_name"}).text

# удалить знаки препинания в признаке в виде текстовых данных
# Определить функцию, которая использует метод translate со словарем знаков препинания
# Загрузить библиотеки
import unicodedata
import sys
# Создать текст
# text_data = ['Hi !!!! I. Love. This. Song....','10000% Agree!!!! #LoveIT','Right?!?!']
# Создать словарь знаков препинания
# punctuation = dict.fromkeys(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))
# Удалить любые знаки препинания во всех строковых значениях
# [string.translate(punctuation) for string in text_data]

# разбить текст на отдельные слова(лексемезировать на слова)
# Загрузить библиотеку
from nltk.tokenize import word_tokenize
# Создать текст
# string = "Сегодняшняя наука — это технология завтрашнего дня"
# Лексемизировать на слова
# word_tokenize(string)

# разбить текст на отдельные слова(лексемизировать текст на предложения)
# Загрузить библиотеку
from nltk.tokenize import sent_tokenize
# Создать текст
# string = """Сегодняшняя наука — это технология завтрашнего дня. Затра начинается сегодня."""
# Лексемизировать на предложения
# sent_tokenize(string)

# удалить чрезвычайно общеупотребительные слова(например, a, is, of, on), несущие в себе мало информации
# Загрузить библиотеку
from nltk.corpus import stopwords
# Перед этим вам следует скачать набор стоп-слов
import nltk
# nltk.download('stopwords')
# Создать лексеьш слов
# tokenized_words = ['i','am','going','to','go','to','the','store','and','park']
# Загрузить стоп-слова
# stop_words_eng = stopwords.words('english')
# stop_words_ru = stopwords.words('russian')
# Удалить стоп-слова
# var = [word for word in tokenized_words if word not in stop_words_eng]

# преобразование лексемизированных слов в их корневые формы(выделение основы слова - корня, и отсечение суффиксов и т.д.)
# Использовать класс Porterstem m er библиотеки NLTK
# Загрузить библиотеку
from nltk.stem.porter import PorterStemmer
# Создать лексемы слов
# tokenized_words = ['i ', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']
# Создать стеммер
# porter = PorterStemmer()
# Применить стеммер
# [porter.stem(word) for word in tokenized_words]
# Для слов русского языка используется стеммер snowball
# Загрузить библиотеку
from nltk.stem.snowball import SnowballStemmer
# Создать лексемы слов
# tokenized_words = ['рыбаки', 'рыбаков', 'рыбаками']
# Создать стеммер
# snowball = SnowballStemmer("russian")
# Применить стеммер
# [snowball.stem(word) for word in tokenized_words]

# собрать лексемизированные слова в синонимические ряды(формы неправильного глагола, из went -> go)
# Загрузить библиотеку
from nltk.stem import WordNetLemmatizer
# Создать лексемы слов
# tokenized_words = ['go', 'went', 'gone', 'am', 'are', 'is', 'was', 'were']
# Создать лемматизатор
# lemmatizer = WordNetLemmatizer()
# Применить лемматизатор
# [lemmatizer.lemmatize(word, pos='v') for word in tokenized_words]

# пометить каждое слово или символ своей частью речи(разметчик частей речи, получится список кортежей со словом и тегом части речи)
# Загрузить библиотеки
from nltk import pos_tag
from nltk import word_tokenize
# Создать текст
# text_data = "Chris loved outdoor running"
# Использовать предварительно натренированный разметчик частей речи
# text_tagged = pos_tag(word_tokenize(text_data))
"""
Метка  Часть речи
NNP    Имя собственное, единственное число
NN     Существительное, единственное число или неисчисляемое
RB     Наречие
VBD    Глагол, прошедшее время
VBG    Глагол, герундий или причастие настоящего времени
JJ     Прилагательное
PRP    Личное местоимение
"""
# Отфильтровать слова по сущесвтительным
# [word for word, tag in text_tagged if tag in ['NN','NNS','NNP','NNPS']]

# наблюдение содержит твит, преобразуем предложения в признаки отдельных частей речи(признак с 1, если присутствует
# собственное существительное 0 в противном случае)
# Загрузить библиотеки
import nltk
from sklearn.preprocessing import MultiLabelBinarizer
# Создать текст
# tweets = ["I am eating a burrito for breakfast","Political science is an amazing field","San Francisco is an awesome city"]
# Создать список
# tagged_tweets = []
# Пометить каждое слово и каждый твит
# for tweet in tweets:
#     tweet_tag = nltk.pos_tag(word_tokenize(tweet))
#     tagged_tweets.append([tag for word, tag in tweet_tag])
# Применить кодирование с одним активным состоянием, чтобы конвертировать метки в признаки
# one_hot_multi = MultiLabelBinarizer()
# one_hot_multi.fit_transform(tagged_tweets)

# создать набор признаков, указывающих на количество вхождений определенного слова в текст наблюдения
# Загрузить библиотеки
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# Создать текст
# text_data = np.array(['Бразилия — моя любовь. Бразилия!','Швеция — лучше','Германия бьет обоих'])
# Создать матрицу признаков на основе мешка слов
# count = CountVectorizer()
# bag_of_words = count.fit_transform(text_data)
# bag_of_words.toarray()
# просмотра слова, связанного с каждым признаком
# Показать имена признаков
# count.get_feature_names()

# мешок слов, но со словами, взвешенными по их важности для наблюдения
# Сравнить частоту слова в документе(твит т.д.) с частотой слова в других документах, используя статистическую меру
# словарной частоты — обратной документной частоты
# Загрузить библиотеки
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# Создать текст
# text_data = np.array(['Бразилия - моя любовь. Бразилия!','Швеция — лучше','Германия бьет обоих'])
# Создать матрицу признаков на основе меры tf-idf
# tfidf = TfidfVectorizer()
# feature_matrix = tfidf.fit_transform(text_data)
# Показать матрицу признаков на основе меры tf-idf как плотную
# feature_matrix.toarray()
# Атрибут vocabuiary_ показывает нам слово каждого признака
# Показать имена признаков
# tfidf.vocabulary_