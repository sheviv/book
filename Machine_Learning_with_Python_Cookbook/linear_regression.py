# ch_13_Линейная регрессия

# натренировать модель, представляющую линейную связь между признаком и вектором целей
# Линейная регрессия
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
# Загрузить данные только с двумя признаками
# boston = load_boston()
# features = boston.data[:, 0:2]
# target = boston.target
# Создать объект линейной регрессии
# regression = LinearRegression()
# Выполнить подгонку линейной регрессии
# model = regression.fit(features, target)
# точка пересечения
# model.intercept_

# Обработка интерактивных эффектов(признак, влияние которого на целевую переменную зависит от другого признака)
# Создать член взаимодействия для захвата этой зависимости
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
# Загрузить данные только с двумя признаками
# boston = load_boston()
# features = boston.data[:, 0:2]
# target = boston.target
# Создать член, характеризующий взаимодействие между признаками
# interaction = PolynomialFeatures(degree=3, include_bias=False, interaction_only=True)
# features_interaction = interaction.fit_transform(features)
# Создать объект линейной регрессии
# regression = LinearRegression()
# Выполнить подгонку линейной регрессии
# model = regression.fit(features_interaction, target)

# Подгонка нелинейной связи(смоделировать нелинейную связь)
# Создать полиномиальную регрессию путем включения полиномиальных признаков в линейную регрессионную модель
# Полиномиальная регрессия — это расширение линейной регрессии, позволяющее моделировать нелинейные связи
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
# Загрузить данные с одним признаком
# boston = load_boston()
# features = boston.data[:,0:1]
# target = boston.target
# Создать полиномиальные признаки х**2 и х**3
# polynomial = PolynomialFeatures(degree=3, include_bias=False)
# degree - максимальное число степеней для поли. признаков
# include_bias=False - удалить признаки, содержащие одни единицы
# features_polynomial = polynomial.fit_transform(features)
# Создать объект линейной регрессии
# regression = LinearRegression()
# Выполнить подгонку линейной регрессии
# model = regression.fit(features_polynomial, target)

# Снижение дисперсии с помощью регуляризации(уменьшить дисперсию линейной регрессионной модели)
# обучающийся алгоритм - сжимающий штраф(регуляризацию), такой как гребневая регрессия и лассо-регрессия
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# boston = load_boston()
# features = boston.data
# target = boston.target
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект гребневой регрессии со значением альфа
# regression = Ridge(alpha=0.5)
# Выполнить подгонку линейной регрессии
# model = regression.fit(features_standardized, target)
# и
# отобрать идеальное значение для гиперпараметра-альфа(на сколько мы штрафуем коэффициенты),
# более высокие значения  - создают более простые модели
from sklearn.linear_model import RidgeCV
# Создать объект гребневой регрессии с тремя значениями alpha
# regr_cv = RidgeCV(alphas=[0.1, 1.0, 10.0])
# Выполнить подгонку линейной регрессии
# model_cv = regr_cv.fit(features_standardized, target)
# Взглянуть на коэффициенты
# model_cv.coef_
# Взглянуть на alpha(наилучшей модели)
# model_cv.alpha_

# уменьшение количества признаков с помощью лассо-регрессии(упростить линейную регрессионную модель,
# уменьшив количество признаков)
from sklearn.linear_model import Lasso
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
# Загрузить данные
# boston = load_boston()
# features = boston.data
# target = boston.target
# Стандартизировать признаки
# scaler = StandardScaler()
# features_standardized = scaler.fit_transform(features)
# Создать объект лассо-регрессии со значением alpha
# regression = Lasso(alpha=0.5)
# Выполнить подгонку линейной регрессии
# model = regression.fit(features_standardized, target)

# лассо-регрессии - может сжимать коэффициенты модели до нуля,
# уменьшая количество признаков в модели(их соответствующие признаки в модели не используются),
# если увеличить а до высокого значения, мы увидим - ни один из признаков не используется.
# можем включить в матрицу признаков 100 признаков, затем путем настройки гиперпараметра а лассо-регрессии создать модель,
# которая использует только(например) 10 наиболее важных признаков. Это позволяет нам уменьшить дисперсию,
# улучшая интерпретируемость модели(меньшее количество признаков легче объяснить)
