
from enum import Enum

class TradingStrategyEnum(Enum):
    LONG_ONLY = 1
    LONG_SHORT_OUTRANGE_MOMEMTUM = 2
    LONG_SHORT_INRANGE_MEAN_REVERSION = 3
    LONG_SHORT_OPPOSITE = 4