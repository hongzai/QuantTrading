from enum import Enum

class ModelEnum(Enum):
    MEAN = "mean"
    ZSCORE = "zscore"
    # EMA = "ema"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    LOG = "log"
    SOFTMAX = "softmax"
    EMA_DIFF = "ema_diff"
    DOUBLE_EMA_CROSSING = "double_ema_crossing"
