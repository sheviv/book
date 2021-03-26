# Базы данных и SQL

# users = [[0, "Hero", 0],
#          [1, "Dunn", 2],
#          [2, "Sue", 3],
#          [3, "Chi", 3]]
# На языке SQL
# СRЕАTE TAВLE users (user_id INT NOT NULL,  # должны быть целыми числами, не может быть NULL
#                     name VARCНAR(200),  # должно быть строкой длиной не более 200 символов
#                     num_friends INT);
# или
# можем вставлять строки с помощью инструкции INSERT:
# INSERT INTO users (user_id, name, num_friends) VALUES (0, 'Hero', 0);
from typing import Tuple, Sequence, List, Any, Callable, Dict, Iterator
from collections import defaultdict
# Несколько псевдонимов типов, которые будут использоваться позже
Row = Dict[str, Any]  # Строка базы данных
WhereClause = Callable[[Row], bool]  # Предикат для единственной строки
HavingClause = Callable[[List[Row]], bool]  # Предикат над многочисленными строками
# нужно передать список имен столбцов и список типов столбцов, как если бы вы создавали таблицу в СУБД SQL:
class Table:
    def __init__(self, columns: List[str], types: List[type]) -> None:
        assert len(columns) == len(types), "число столбцов должно == числу типов"
        self.columns = columns  # Имена столбцов
        self.types = types  # Типы данных в столбцах
        self.rows: List[Row] = []  # (данньх пока нет)
    # Мы добавим вспомогательный метод, который будет получать тип столбца:
    def col2type(self, col: str) -> type:
        idx = self.columns.index(col)  # Отыскать индекс столбца
        return self.types[idx]  # и вернуть его тип
    # insert, который проверяет допустимость вставляемых значений, необходимо указать правильное число значений,
    # и каждое должно иметь правильный тип (или None):
    def insert(self, values: list) -> None:
        # Проверить, правильное ли число значений
        if len(values) != len(self.types):
            raise ValueError(f"Tpeбyeтcя {len(self.types)} значений")
            # Проверить допуститмость типов значений
        for value, typ3 in zip(values, self.types):
            if not isinstance(value, typ3) and value is not None:
                raise TypeError(f"Ожидаемый тип {typ3}, но получено {value}")
        # Добавить соответствующий словарь как "строку"
        self.rows.append(dict(zip(self.columns, values)))
    # несколько дандерных методов(для тестирования нашего кода)
    def __getitem__(self, idx: int) -> Row:
        return self.rows[idx]
    def __iter__(self) -> Iterator[Row]:
        return iter(self.rows)
    def __len__(self) -> int:
        return len(self.rows)
    # метод структурированной печати нашей таблицы
    def __repr__(self):
        """Структурированное представление таблицы: столбцы затем строки"""
        rows = "\n".join(str(row) for row in self.rows)
        return f"{self.columns}\n{rows}"
# создать таблицу Users:
# Конструктор требует имена и типы столбцов
users = Table(['user_id', 'name', 'num_friends'], [int, str, int])
users.insert([0, "Hero", 0])
users.insert([1, "Dunn", 2])


# Инструкция UPDAТЕ
# обновить данные, которые уже есть в базе данных
# если Dunn приобретает еще одного друга
# UPDATE users
# SET num_friends = 3
# user_id = 1;

# dict, ключи - столбцы для обновления,а значения новые значения полей. Вторым аргументом - функция predicate(возвращает
# True для строк, которые должны быть обновлены, и False)
def update(self, updates: Dict[str, Any], predicate: WhereClause = lambda row: True):
    # Сначала убедиться, что обновления имеют допустимые имена и типы
    for colurnn, new_value in updates.items():
        if colurnn not in self.colurnns:
            raise ValueError (f"недопустимый столбец: {colurnn}")
        typ3 = self.col2type(colurnn)
        if not isinstance(new_value, typ3) and new_value is not None:
            raise TypeError(f"ожидаемый тип {typ3}, но получено {new_value}")
    # Теперь обновить
    for row in self.rows:
        if predicate(row):
            for colurnn, new_value in updates.items():
                row[colurnn] = new_value
assert users[1]['num_friends'] == 2  # Исходное значение
users.update({'num_friends' : 3},lambda row: row['user _ id'] == 1)  # Назначить num_friends = З,где user_id == 1
assert users[1]['num_friends'] == 3  # Обновить значение


# Инструкция DELETE
# удаления строк из таблицы в SQL. Опасный способ удаляет все строки таблицы:
# DELEТE FROM users;
# Менее опасный - добавляет условное выражение WHERE и удаляет те строки, которые удовлетворяют определенному условию:
# DELEТE FROM users WHERE user_id = 1;
# добавить этот функционал в нашу таблицу Table:
def delete(self, predicate: WhereClause = lambda row: True) -> None:
    """Удалить все строки, совпадающие с предикатом"""
    self.rows = [row for row in self.rows if not predicate(row)]
# # Мы на самом деле не собираемся использовать эти строки кода
users.delete(lambda row: row["user_id"] == 1)  # Удаляет строки с user_id 1
users.delete()  # Удаляет все строки


# Инструкция SELECT
# таблицы SQL целиком не инспектируются. из них выбирают данные с помощью инструкции SELECT:
# SELECT * FROM users;  # Получить все содержимое
# SELECT * FROM users LIMIT 2;  # Получить первые две строки
# SELECТ user_id FROM users;  # Получить только конкретные столбцы
# SELECТ user_id FROM users WHERE name = 'Dunn';  # Получить конкретные строки

# инструкцию SELECT для вычисления полей:
# SELECТ LENGHT(name) AS name_length FROM users;

# метод select, который возвращает новую таблицу(метод принимает два необязательных аргумента:
# keep_colurnns - задает имена столбцов, которые хотите сохранить в результирующей таблице.
# Если он не указан - результат будет содержать все столбцы;
# additional_columns - словарь, ключи - новые имена столбцов,
# значения - функции, определяющими порядок вычисления значений новых столбцов
# Если не предоставите ни один из них, то просто получите копию таблицы:

def select(self, keep_columns: List[str] = None, additional_columns: Dict[str, Callable] = None) -> 'Table':
    if keep_columns is None:  # Если ни один столбец не указан,
        keep_columns = self.columns  # то вернуть все столбцы
    if additional_columns is None:
        additional_columns = {}
    # Имена и типы новых столбцов
    new_columns = keep_columns + list(additional_columns.keys())
    keep_types = [self.col2type(col) for col in additional_columns]
    # Вот как получить возвращаемый тип из аннотации типа.
    # Это даст сбой, если "вычисление" не имеет возвращаемого типа
    add_types = [calculation.__annotations__['return'] for calculation in additional_columns.values()]
    # Создать новую таблицу для результатов
    new_tаblе = Table(new_columns, keep_types + add_types)
    for row in self.rows:
        new_row = [row[column] for column in keep_columns]
        for column_name, calculation in additional_columns.items():
            new_row.append(calculation(row))
        new_tаblе.insert(new_row)
    return new_tаblе


# Инструкция GROUP ВУ
# GROUP ВУ - группирует строки с одинаковыми значениями в указанных полях, применяет агрегатные функции(MIN, МАХ, COUNT и suм)
# отыскать число пользователей и наименьший идентификатор user_id для каждой возможной длины имени:
# SELECT LENGТН(name) AS name_length,
#     МIN(user_id) AS min_user_id,
#     СООNТ(*) AS num_users
# FROM users
# GROOP users
# GROOP ВУ LENGHT(name);

# узнать среднее число друзей для пользователей, чьи имена начинаются с определенных букв,
# но увидеть только результаты для тех букв, чье соответствующее среднее больше 1.
# SELECТ SUВSТR(name, 1, 1) AS first_letter,
#     AVG(num_friends) AS avg_num_friends
# FROM users
# GROOP ВУ SUВSТR(name, 1, 1)
# HAVING AVG(nurn_friends) > 1;  # фильтр применяется к агрегатам

# вычислять совокупные агрегатные величины - опускаем выражение GROUP_ВУ:
# SELECТ SOM(user_id) AS user_id_sum
# FROM users
# WHERE user id > 1;


# Инструкция ORDER ВУ
# отсортировать результаты. знать первые два имени ваших пользователей(в алфавитном порядке):
# SELECТ * FROM users
# ORDER ВУ name
# LIМIT 2;


# Инструкция JOIN
# Она сочетает строки в левой таблице с соответствующими строками в правой таблице, значение - слова "соответствующий"
# зависит от того, как задается объединение.
# SELECТ users.name
# FROM users
# JOIN user_interests
# CN users.user_id = user_interests.user_id
# WHERE user_interests.interest = 'SQL'
# для каждой строки в таблице users нам следует обратиться к идентификатору user_id и ассоциировать эту строку
# с каждой строкой в таблице user_interests, содержащей тот же самый идентификатор user_id.
# какие таблицы объединять(JOIN) и по каким столбцам(ON).

# Используя инструкцию LEFT JOIN - подсчитать число интересов у каждого пользователя:
# SELECТ users.id, COUNТ(user_interests.interest) AS num_interests
# FROM users
# LEFT JOIN user_interests
# CN users.user_id = user_interests.user_id


# Подзапросы
# выборку (инструкцией SELECT) из (объединенных инструкцией JOIN) результатов запросов, как если бы они были таблицами.
# SELECТ MIN(user_id) AS min_user_id FROM
# (SELECТ user_id FROM user_interests WHERE interest = 'SQL') sql_interests;


# Индексы
# Каждая таблица в базе данных может иметь один или несколько индексов, которые позволяют выполнять
# быстрый просмотр таблиц по ключевым столбцам, эффективно объединять таблицы и накладывать уникальные ограничения
# на столбцы или их сочетания.

