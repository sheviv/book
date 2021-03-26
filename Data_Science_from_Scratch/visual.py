from matplotlib import pyplot as plt

# years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]  # Годы
# gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]  # ВВП
# # Создать линейную диаграмму: годы по оси х, ВВП по оси у
# plt.plot(years, gdp, color='green', marker='o', linestyle='solid')
# # Добавить название диаграммы
# plt.title("Номинальный ВВП")
# # Добавить подпись к оси у, x
# plt.ylabel("Мpд $")
# plt.xlabel("Год")
# plt.show()

# Столбчатые графики
# movies = ["Энни Холл", "Бен-Гур", "Касабланка", "Ганди", "Вестсайдская история"]
# num_oscars = [5, 11, 3, 8, 10]
# # Построить столбцы с левыми Х-координатами [xs] и высотами [num_oscars]
# plt.bar(range(len(movies)), num_oscars)
# # Добавить заголовок
# plt.title("Moи любимые фильмы")
# # Разместить метку на оси у
# plt.ylabel("Количество наград")
# # Пометить ось х названиями фильмов в центре каждого столбца
# plt.xticks(range(len(movies)), movies)
# plt.show()

from collections import Counter
# grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]
# # Сгруппировать оценки nодецильно, но
# # разместить 100 вместе с отметками 90 и вьnnе
# histograrn = Counter(min(grade // 10 * 10, 90) for grade in grades)
# plt.bar([x + 5 for x in histograrn.keys()],  # Сдвинуть столбец влево на 5
#         list(histograrn.values()),  # Высота столбца
#         10)  # Ширина каждого столбца 10
# plt.axis([-5, 105, 0, 5])  # Ось х от -5 до 105,
# # ОСЬ у ОТ O ДО 5
# plt.xticks([10 * i for i in range(11)])  # Метки по оси х: О, 10, ... , 100
# plt.xlabel("Дециль")
# plt.ylabel("Число студентов")
# plt.title("Pacnpeдeлeниe оценок за экзамен No 1")
# plt.show()

# mentions = [500, 505]  # Упоминания
# years = [2017, 2018]  # Годы
# plt.bar(years, mentions, 0.8)
# plt.xticks(years)
# plt.ylabel("Число раз, когда я слышал, как упоминали науку о данных")
# # Если этого не сделать, matplotlib подпишет ось х как О, 1 и добавит +2.ОlЗеЗ в правом углу (недоработка matplotlib!)
# plt.ticklabel_format(useOffset=False)
# # Дезориентирующая ось у показывает только то, что выше 500
# plt.axis([2016.5, 2018.5, 499, 506])
# plt.title("Пocмoтpитe, какой 'огромный' прирост 1 ")
# plt.show()

# Линейные графики
# variance = [1, 2, 4, 8, 16, 32, 64, 128, 256] # Дисперсия
# bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1] # Квадрат смещения
# # Суммарная ошибка
# total_error = [x + y for x, y in zip(variance, bias_squared)]
# xs = [i for i, _ in enumerate(variance)]
# # Метод plt.plot можно вызывать много раз, чтобы показать несколько графиков на одной и той же диаграмме:
# # зеленая сплошная линия
# plt.plot(xs, variance, 'g-', label='дисперсия')
# # красная штрихпунктирная
# plt.plot(xs, bias_squared, 'r-', label='смещение^2')
# # синяя пунктирная
# plt.plot(xs, total_error, 'b:', label='суммарная ошибка')
# # Ее.ли для каждой линии задано название label, то легенда будет показана автоматически,
# # loc=9 означает "наверху посередине"
# plt.legend(loc=9)
# plt.xlabel("Сложность модели")
# plt.title("Koмпpoмиcc между смещением и дисперсией")
# plt.show()

# Диаграммы рассеяния
# friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67] # Друзья
# minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190] # Минуты
# labels = ['а', 'b', 'с', 'd', 'е', 'f', 'g', 'h', 'i'] # Метки
# plt.scatter(friends, minutes)
# # Назначить метку для каждой точки
# for label, friend_count, minute_count in zip(labels, friends, minutes):
#     plt.annotate(label,
#                  xy=(friend_count, minute_count),  # Задать метку
#                  xytext=(5, -5),
#                  textcoords='offset points')
# plt.title("Чиcлo минут против числа друзей")
# plt.xlabel("Число друзей")
# plt.ylabel("Чиcлo минут, проводимых на сайте ежедневно")
# plt.axis( "equal"),
# plt.show()
