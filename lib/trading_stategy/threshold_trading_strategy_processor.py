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
        
        # > up threshold = long | otherwise = no position
        if self.trading_strategy == ThresholdTradingStrategyEnum.LONG_ABOVE_UPPER:
            position[data[alpha_col] > data[upper_threshold_col]] = 1
        # < low threshold = short | otherwise = no position    
        elif self.trading_strategy == ThresholdTradingStrategyEnum.SHORT_BELOW_LOWER:
            position[data[alpha_col] < data[lower_threshold_col]] = -1
        # > up threshold = long | < low threshold = short | in range = no position    
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM:
            position[data[alpha_col] > data[upper_threshold_col]] = 1
            position[data[alpha_col] < data[lower_threshold_col]] = -1
        # > up threshold = short | < low threshold = long | in range = no position    
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM_REVERSE:
            position[data[alpha_col] > data[upper_threshold_col]] = -1
            position[data[alpha_col] < data[lower_threshold_col]] = 1
        # > up threshold = short | < low threshold = long | in range = no position    
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_INRANGE_MEAN_REVERSION:
            position[data[alpha_col] > data[upper_threshold_col]] = -1
            position[data[alpha_col] < data[lower_threshold_col]] = 1
        # > up threshold = long | < low threshold = short | in range = no position    
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_INRANGE_MEAN_REVERSION_REVERSE:
            position[data[alpha_col] > data[upper_threshold_col]] = 1
            position[data[alpha_col] < data[lower_threshold_col]] = -1
        # > up threshold = long | < low threshold = short | in range = keep previous position    
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
        # > up threshold = short | < low threshold = long | in range = keep previous position    
        elif self.trading_strategy == ThresholdTradingStrategyEnum.LONG_SHORT_OPPOSITE_REVERSE:
            for i in range(0, len(data)):
                previous_position = position[i-1]
                alpha_value = data[alpha_col].iloc[i]
                lower_threshold = data[lower_threshold_col].iloc[i]
                upper_threshold = data[upper_threshold_col].iloc[i]
                if (previous_position == 1 and alpha_value <= lower_threshold) or (previous_position == -1 and alpha_value > upper_threshold):
                    position[i] = previous_position
                elif alpha_value < lower_threshold:
                    position[i] = 1
                elif alpha_value > upper_threshold:
                    position[i] = -1
                else:
                    position[i] = previous_position

        elif self.trading_strategy == ThresholdTradingStrategyEnum.DOUBLE_BOLLINGER:
            for i in range(0, len(data)):
                alpha_value = data[alpha_col].iloc[i]
                lower_1 = data[f'{lower_threshold_col}_1'].iloc[i] 
                upper_1 = data[f'{upper_threshold_col}_1'].iloc[i] 
                lower_2 = data[f'{lower_threshold_col}_2'].iloc[i] 
                upper_2 = data[f'{upper_threshold_col}_2'].iloc[i]  
                
        
                if alpha_value < lower_1 and alpha_value > lower_2:
                    position[i] = 1 
                elif alpha_value > upper_1 and alpha_value < upper_2:
                    position[i] = -1 
                else:
                    position[i] = position[i-1] if i > 0 else 0           

        return position
    # ----- End trade -----
