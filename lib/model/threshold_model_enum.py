from enum import Enum

class ThresholdModelEnum(Enum):
    MA = "ma"
    EMA = "ema"
    ZSCORE = "zscore"
    MA_DIFF = "ma_diff"
    EMA_DIFF = "ema_diff"
    MINMAX = "minmax"
    ROBUST = "robust"
    RSI = "rsi"
    LINEAR_REGRESSION = "linear_regression"
    PERCENTILE = "percentile"
    BOLLINGER = 'bollinger'
    EMA_BOLLINGER = "ema_bollinger"
    DOUBLE_BOLLINGER = "double_bollinger"
    
    #MAXABS = "maxabs"
    #LOG = "log"
    #SOFTMAX = "softmax"
    #DOUBLE_EMA_CROSSING = "double_ema_crossing"