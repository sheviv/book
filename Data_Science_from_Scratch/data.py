# именованных кортежей namedtuple, который похож на кортеж, но с именованными слотами:
from collections import namedtuple
import datetime
# StockPrice = namedtuple( 'StockPrice', [ 'syrnЬol', 'date', 'closing_price'])
# price = StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)
# кортежи narnedtuple немутируемы

# Классы данных dataclasses
from dataclasses import dataclass
# @dataclass
# class StockPrice2:
#     symbol: str
#     date: datetime.date
#     closing_price: float
#     def is_high_tech(self) -> bool:
#         """Это класс, и поэтому мы также можем добавлять методы"""
#         return self.symbol in ['MSFГ', 'GOOG', 'FВ', 'AМZN', 'AAPL']
# price2 = StockPrice2('MSFГ', datetime.date(2018, 12, 14), 106.03)

# Очистка и конвертирование
from dateutil.parser import parse
# def parse_row(row: List[str]) -> StockPrice:
#     symbol, date, closing_price = row
#     return StockPrice(symbol=symbol, date=parse(date) .date(), closing_price=float(closing_price))
# # Теперь протестируем нашу функцию
# stock = parse_row(["MSFT'", "2018-12-14", "106.03"])

import tqdm
import random
# for i in tqdm.tqdm(range(100)):
#     # Делать что-то медленное
#     _ = [random.random() for _ in range(1000000)]
