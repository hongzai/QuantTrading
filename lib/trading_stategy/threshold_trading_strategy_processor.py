import pandas as pd
import numpy as np
from lib.trading_stategy.threshold_trading_strategy_enum import ThresholdTradingStrategyEnum

class ThresholdTradingStrategyProcessor:
    def __init__(self, trading_strategy: ThresholdTradingStrategyEnum):
        self.trading_strategy = trading_strategy

    # ----- Begin trade -----
    def run(self, data: pd.DataFrame, alpha_column_name: str, lower_threshold_col: str, upper_threshold_col: str):
        position = np.zeros(len(data))
        alpha_col = alpha_column_name
        
        if self.trading_strategy == ThresholdTradingStrategyEnum.LONG_ABOVE_UPPER:
            position[data[alpha_col] > data[upper_threshold_col]] = 1
        elif self.trading_strategy == ThresholdTradingStrategyEnum.SHORT_BELOW_LOWER:
            position[data[alpha_col] < data[lower_threshold_col]] = -1
        if self.trading_strategy == ThresholdTradingStrategyEnum.LONG_ABOVE_UPPER:
            position[data[alpha_col] > data[upper_threshold_col]] = 1
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM:
            position[data[alpha_col] > data[upper_threshold_col]] = 1
            position[data[alpha_col] < data[lower_threshold_col]] = -1
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_INRANGE_MEAN_REVERSION:
            position[data[alpha_col] > data[upper_threshold_col]] = -1
            position[data[alpha_col] < data[lower_threshold_col]] = 1
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_OPPOSITE:
            for i in range(0, len(data)):
                previous_position = position[i-1]
                alpha_value = data[alpha_col].iloc[i]
                lower_threshold = data[lower_threshold_col].iloc[i]
                upper_threshold = data[upper_threshold_col].iloc[i]
                if (previous_position == -1 and alpha_value <= lower_threshold) or (previous_position == 1 and alpha_value > upper_threshold):
                    position[i] = previous_position
                elif alpha_value < lower_threshold:
                    position[i] = -1
                elif alpha_value > upper_threshold:
                    position[i] = 1
                else:
                    position[i] = previous_position

        return position
    # ----- End trade -----
