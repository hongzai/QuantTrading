from enum import Enum

class AlphaCombineMethodEnum(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply" 
    DIVIDE = "divide" 
    DIVIDE_INVERSE = "divide_inverse" 
    WEIGHTED_ADD = "weighted_add" 
    LOG_RATIO = "log_ratio" 
    PERCENT_DIFF = "percent_diff" 
    GEOMETRIC_MEAN = "geometric_mean" 
    HARMONIC_MEAN = "harmonic_mean" 