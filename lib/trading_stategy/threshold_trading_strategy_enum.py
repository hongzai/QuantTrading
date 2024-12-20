from enum import Enum

class ThresholdTradingStrategyEnum(Enum):
    LONG_ABOVE_UPPER = 1
    SHORT_BELOW_LOWER = 2
    LONG_SHORT_OUTRANGE_MOMEMTUM = 3
    LONG_SHORT_INRANGE_MEAN_REVERSION = 4
    LONG_SHORT_OPPOSITE = 5
    LONG_SHORT_OPPOSITE_REVERSE = 6
    LONG_SHORT_OUTRANGE_MOMEMTUM_REVERSE = 7
    LONG_SHORT_INRANGE_MEAN_REVERSION_REVERSE = 8
    DOUBLE_BOLLINGER = 9