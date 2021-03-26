# Использование интерфейсов API

# Форматы JSON и XML
# import json
# serialized = """{ "title" : "Data Science Book",
# "author" : "Joel Grus",
# "puЫicationYear" : 2019,
# "topics" : [ "data", "science", "data science"] 1 """
# Разобрать JSON, создав Руthоn'овск:ий словарь
# deserialized = json.loads(serialized)

# Использование неаутентифицированного API
# import requests, json
# github_user = "joelgrus"
# endpoint = f"https://api.github.com/users/{github_user}/repos"
# repos = json.loads(requests.get(endpoint).text)
# # список из словарей, каждый из которых обозначает публичное хранилище в учетной записи GitHub
# from collections import Counter
# from dateutil.parser import parse
# dates = [parse(repo["created_at"]) for repo in repos]  # список дат
# month_counts = Counter(date.month for date in dates)  # число месяцев
# weekday_counts = Counter(date.weekday() for date in dates)  # число будних дней
# # Схожим образом вы можете получить языки программирования моих последних пяти хранилищ:
# last_5_repositories = sorted(repos, # последние 5 хранилищ
#                             key=lambda r: r["created_at"],
#                             reverse=True)[:5]
# last_5_languages = [repo["language"]  # последние 5 языков
#                     for repo in last_5_repositories]

# Пример: использование АРI-интерфейсов Twitter
import os
# Вы можете свободно встроить свой ключ и секрет прямиком в код
# CONSUМER_KEY = os.environ.get("КЛЮЧ_ПОТРЕБИТЕЛЯ_TWITTER")
# CONSUМER_SECRET = os.environ.get("СЕКРЕТ_ПОТРЕБИТЕЛЯ_TWITTER")
# Теперь мы можем инстанцировать клиента:
import webbrowser
from twython import Twython
# Получить временного клиента для извлечения аутентификационного URL
# temp_client = Twython(CONSUМER_KEY, CONSUМER_SECRET)
# temp_creds = temp_client.get_authentication_tokens()
# url = temp_creds['auth_url']
# # Теперь посетить этот URL для авторизации приложения и получения PIN
# print(f"перейдите на {url} и получите РIN-код и вставьте его ниже")
# webbrowser.open(url)
# PIN_CODE = input("пожалуйста, введите РIN-код: ")
# # Теперь мы используем PIN_CODE для получения фактических токенов
# auth_client = Twython(CONSUМER_KEY,
#                       CONSUМER_SECRET,
#                       temp_creds['oauth_token'],
#                       temp_creds['oauth_token secret'])
# final_step = auth_client.get_authorized_tokens(PIN_CODE)
# ACCESS_TOKEN = final_step['oauth_token']
# ACCESS_TOKEN_SECRET = final_step['oauth_token_secret']
# # И получим новый экземпляр Twython, используя их.
# twitter = Twython(CONSUМER_KEY,
#                   CONSUМER_SECRET,
#                   ACCESS_TOKEN,
#                   ACCESS_TOKEN_SECRET)
# # Найти твиты, содержащие фразу "data science"
# for status in twitter.search(q="'data science'")["statuses"]:
#     user = status["user"]["screen_name"]
#     text = status["text"]
#     print(f"{user}:{text}\n")

# Streaming - интерфейс позволяет подключаться к огромному потоку сообщений Twitter
# определить класс, который наследует у класса тwythonStreamer и переопределяет его метод on_success и метод on_error:
from twython import TwythonStreamer
# Добавлять данные в глобальную переменную - это пример плохого стиля, но он намного упрощает пример
# tweets = []
# class MyStreamer(TwythonStreamer):
#     """Наш собственный подкласс КJJacca TwythonStreamer, который
#     определяет, как взаимодействовать с потоком"""
#     def on_success(self, data):
#         """Что делать, когда Twitter присылает данные?
#         Здесь данные будут в виде словаря, представляющего твит"""
#         # Мы хотим собирать твиты только на английском
#         if data['lang'] == 'ел':
#             tweets.append(data)
#             print("received tweet #", len(tweets))
#         # Остановиться, когда собрано достаточно
#         if len(tweets) >= 1000:
#             self.disconnect()
#     def on_error(self, status_code, data):
#         print(status_code, data)
#         self.disconnect()
# Подкласс MyStreamer подКJJючится к потоку Twitter и будет ждать, когда Twitter подаст данные.
# Осталось только инициализировать его и выполнить:
# stream = MyStreamer(CONSUМER_KEY, CONSUМER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
# Начинает потреблять публичные новостные ленты,
# которые содержат ключевое слово 'data'
# stream.statuses.filter(track='data')
# Если же, напротив, мы хотим начать потреблять
# ВСЕ публичные новостные ленты
# stream.statuses.sample()

# Ведущие хештеги
from collections import Counter
# top_hashtags = Counter(hashtag['text'].lower()
#                        for tweet in tweets
#                        for hashtag in tweet["entities"]["hashtags"])
# print(top_hashtags.most_common(5))
