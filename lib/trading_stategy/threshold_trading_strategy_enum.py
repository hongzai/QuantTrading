
from enum import Enum

class ThresholdTradingStrategyEnum(Enum):
    LONG_ABOVE_UPPER = 1
    SHORT_BELOW_LOWER = 2
    LONG_SHORT_OUTRANGE_MOMEMTUM = 3
    LONG_SHORT_INRANGE_MEAN_REVERSION = 4
    LONG_SHORT_OPPOSITE = 5