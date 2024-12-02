import decimal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lib.model.threshold_model_enum import ThresholdModelEnum

'''
To model alpha for signal generation
'''
class ThresholdModelProcessor:
    def __init__(self, model: ThresholdModelEnum):
        self.model = model

    def run(self, data: pd.DataFrame, column_name: str, lower_threshold_col: str, upper_threshold_col: str, rolling_window: int, diff_threshold: decimal):
        if self.model == ThresholdModelEnum.MA:
            # Calculate sma for specific column
            # Create upperbound and lowerbound based on sma and diff_threshold
            # Use ORIGINAL column as alpha against threshold
            ma_column_name = f'{column_name}-ma'
            data[ma_column_name] = self.calculate_ma(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[lower_threshold_col], data[upper_threshold_col] = self.calculate_thresholds(data[ma_column_name], diff_threshold)
            
            return column_name
        
        elif self.model == ThresholdModelEnum.MA_REVERSE:
            # Calculate sma for specific column
            # Create upperbound and lowerbound based on sma and diff_threshold
            # Use ORIGINAL column as alpha against threshold for reverse signals
            ma_column_name = f'{column_name}-ma'
            data[ma_column_name] = self.calculate_ma(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[lower_threshold_col], data[upper_threshold_col] = self.calculate_thresholds(data[ma_column_name], diff_threshold)
            
            return column_name

        if self.model == ThresholdModelEnum.EMA:
            # Calculate sma for specific column
            # Create upperbound and lowerbound based on sma and diff_threshold
            # Use ORIGINAL column as alpha against threshold
            ema_column_name = f'{column_name}-ema'
            data[ema_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[lower_threshold_col], data[upper_threshold_col] = self.calculate_thresholds(data[ema_column_name], diff_threshold)
            
            return column_name
        
        elif self.model == ThresholdModelEnum.ZSCORE:
            # Calculating zscore for specific column
            # Create upperbound and lowerbound based on diff_threshold
            # Use zscore as alpha against threshold
            zscore_column_name = f'{column_name}-zscore'
            data[zscore_column_name] = self.calculate_zscore(data, column_name, rolling_window)
            
            data[lower_threshold_col] = -diff_threshold if diff_threshold > 0 else diff_threshold * 2
            data[upper_threshold_col] = diff_threshold

            return zscore_column_name
        
        elif self.model == ThresholdModelEnum.MA_DIFF: 
            # Calculating change percentage zscore of specific column
            # Create upperbound and lowerbound based on diff_threshold
            # Use ma_diff column as alpha against threshold
            ma_column_name = f'{column_name}-ma'
            data[ma_column_name] = self.calculate_ma(data, column_name, rolling_window)
            
            ma_diff_column_name = f'{column_name}-ma_diff'
            data[ma_diff_column_name] = data[column_name] - data[ma_column_name]
            
            data[lower_threshold_col] = -diff_threshold
            data[upper_threshold_col] = diff_threshold

            return ma_diff_column_name
        
        elif self.model == ThresholdModelEnum.EMA_DIFF: 
            # Calculating change percentage zscore of specific column
            # Create upperbound and lowerbound based on diff_threshold
            # Use ema_diff column as alpha against threshold
            ema_column_name = f'{column_name}-ema'
            data[ema_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            ema_diff_column_name = f'{column_name}-ema_diff'
            data[ema_diff_column_name] = data[column_name] - data[ema_column_name]
            
            data[lower_threshold_col] = -diff_threshold
            data[upper_threshold_col] = diff_threshold

            return ema_diff_column_name
        
        elif self.model == ThresholdModelEnum.EMA_REVERSE:
            # Calculate ema for specific column
            # Create upperbound and lowerbound based on ema and diff_threshold
            # Use ORIGINAL column as alpha against threshold for reverse signals
            ema_column_name = f'{column_name}-ema'
            data[ema_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[lower_threshold_col], data[upper_threshold_col] = self.calculate_thresholds(data[ema_column_name], diff_threshold)
            
            return column_name

        elif self.model == ThresholdModelEnum.MINMAX:
            # Min-Max Normalization
            minmax_column_name = f'{column_name}-minmax'
            data[minmax_column_name] = self.calculate_minmax(data, column_name, rolling_window)
            
            # Min-Max thresholds typically in range [-1, 1]
            data[lower_threshold_col] = -diff_threshold if diff_threshold > 0 else diff_threshold * 2
            data[upper_threshold_col] = diff_threshold

            return minmax_column_name

        elif self.model == ThresholdModelEnum.RSI:
            # Calculate RSI using the rolling window as period
            rsi_column_name = f'{column_name}-rsi'
            data[rsi_column_name] = self.calculate_rsi(data, column_name, rolling_window)
            
            data[lower_threshold_col] = diff_threshold
            data[upper_threshold_col] = 100 - diff_threshold
            
            print(f"[{self.__class__.__name__}] RSI Period={rolling_window}, Thresholds={data[lower_threshold_col].iloc[-1]}/{data[upper_threshold_col].iloc[-1]}")
            
            return rsi_column_name

        elif self.model == ThresholdModelEnum.LINEAR_REGRESSION:
            regression_column_name = f'{column_name}-regression'
            data[regression_column_name] = self.calculate_linear_regression(data, column_name, rolling_window)
            
            data[lower_threshold_col] = data[regression_column_name] - diff_threshold
            data[upper_threshold_col] = data[regression_column_name] + diff_threshold
            
            return regression_column_name

        elif self.model == ThresholdModelEnum.PERCENTILE:
            percentile_column_name = f'{column_name}-percentile'
            data[percentile_column_name] = self.calculate_percentile(data, column_name, rolling_window)
            
            data[lower_threshold_col] = data[percentile_column_name].quantile(0.25)
            data[upper_threshold_col] = data[percentile_column_name].quantile(0.75)
            
            return percentile_column_name
        
        elif self.model == ThresholdModelEnum.ROBUST:
            robust_column_name = f'{column_name}-robust'
            data[robust_column_name] = self.calculate_robust(data, column_name, rolling_window)
            
            # Thresholds for robust normalization
            data[lower_threshold_col] = -diff_threshold if diff_threshold > 0 else diff_threshold * 2
            data[upper_threshold_col] = diff_threshold
            
            return robust_column_name

        elif self.model == ThresholdModelEnum.BOLLINGER:
            bb_column_name = f'{column_name}-bb'
            data[bb_column_name] = self.calculate_ma(data, column_name, rolling_window)
            
            std = data[column_name].rolling(window=rolling_window).std()
            
            data[lower_threshold_col] = data[bb_column_name] - (std * diff_threshold)
            data[upper_threshold_col] = data[bb_column_name] + (std * diff_threshold)
            
            return column_name

        elif self.model == ThresholdModelEnum.BOLLINGER_REVERSE:
            bb_column_name = f'{column_name}-bb'
            data[bb_column_name] = self.calculate_ma(data, column_name, rolling_window)
            
            std = data[column_name].rolling(window=rolling_window).std()
            
            data[lower_threshold_col] = data[bb_column_name] + (std * diff_threshold)
            data[upper_threshold_col] = data[bb_column_name] - (std * diff_threshold)
            
            return column_name

        elif self.model == ThresholdModelEnum.EMA_BOLLINGER:
            bb_column_name = f'{column_name}-ema-bb'
            data[bb_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            # calculate standard deviation using EMA
            squared_diff = (data[column_name] - data[bb_column_name]) ** 2
            squared_diff_df = pd.DataFrame(squared_diff, columns=[squared_diff.name])
            ema_std = np.sqrt(self.calculate_ema(squared_diff_df, squared_diff.name, rolling_window))
            
            data[lower_threshold_col] = data[bb_column_name] - (ema_std * diff_threshold)
            data[upper_threshold_col] = data[bb_column_name] + (ema_std * diff_threshold)
            
            return column_name

        elif self.model == ThresholdModelEnum.EMA_BOLLINGER_REVERSE:
            bb_column_name = f'{column_name}-ema-bb'
            data[bb_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            # calculate standard deviation using EMA
            squared_diff = (data[column_name] - data[bb_column_name]) ** 2
            squared_diff_df = pd.DataFrame(squared_diff, columns=[squared_diff.name])
            ema_std = np.sqrt(self.calculate_ema(squared_diff_df, squared_diff.name, rolling_window))
                
            # reverse thresholds
            data[lower_threshold_col] = data[bb_column_name] + (ema_std * diff_threshold)
            data[upper_threshold_col] = data[bb_column_name] - (ema_std * diff_threshold)
            
            return column_name
       

    # ----- Begin Model Formula-----
    def calculate_thresholds(self, targets: pd.Series, diff_threshold: decimal):
        upper_thresholds = targets * (1.0 + diff_threshold)
        lower_thresholds = targets * (1.0 - diff_threshold)

        negative_mask = targets < 0
        upper_thresholds[negative_mask] = targets[negative_mask] * (1.0 - diff_threshold)
        lower_thresholds[negative_mask] = targets[negative_mask] * (1.0 + diff_threshold)
        
        return lower_thresholds, upper_thresholds

    def calculate_zscore(self, data: pd.DataFrame, column_name: str, rolling_window: decimal):
        data['rolling_mean'] = data[column_name].rolling(rolling_window).mean()
        data['rolling_std'] = data[column_name].rolling(rolling_window).std()

        return (data[column_name] - data['rolling_mean']) / data['rolling_std']
    
    def calculate_ma(self, data: pd.DataFrame, column_name: str, rolling_window: decimal):
        return data[column_name].rolling(rolling_window).mean()
    
    def calculate_ema(self, data, column_name, span):
        return data[column_name].ewm(span=span, adjust=False).mean()
    
    def calculate_minmax(self, data: pd.DataFrame, column_name: str, rolling_window: int):
        data['rolling_min'] = data[column_name].rolling(rolling_window).min()
        data['rolling_max'] = data[column_name].rolling(rolling_window).max()
        
        # Avoid division by zero
        range_diff = data['rolling_max'] - data['rolling_min']
        range_diff = range_diff.replace(0, 1e-8)
        
        # Apply Min-Max normalization to range [-1, 1]
        return 2 * (data[column_name] - data['rolling_min']) / range_diff - 1
            
    def calculate_linear_regression(self, data: pd.DataFrame, column_name: str, rolling_window: int):
        def linear_regression(window):
            if len(window) < rolling_window:
                return np.nan  # Handle cases where window size is less than rolling_window
            x = np.arange(len(window))
            y = window.values
            slope, intercept = np.polyfit(x, y, 1)
            return slope * x[-1] + intercept
        
        return data[column_name].rolling(window=rolling_window).apply(linear_regression, raw=False)
    
    def calculate_rsi(self, data: pd.DataFrame, column_name: str, rolling_window: int):
        # Calculate price changes
        delta = data[column_name].diff()
        
        # Get gains (positive) and losses (negative)
        gain = (delta.where(delta > 0, 0))
        loss = (-delta.where(delta < 0, 0))
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=rolling_window).mean()
        avg_loss = loss.rolling(window=rolling_window).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        return 100 - (100 / (1 + rs))
        
    def calculate_percentile(self, data: pd.DataFrame, column_name: str, rolling_window: int):
        return data[column_name].rolling(window=rolling_window).apply(lambda x: np.percentile(x, 50), raw=False)
    
    def calculate_robust(self, data: pd.DataFrame, column_name: str, rolling_window: int) -> pd.Series:
        median = data[column_name].rolling(window=rolling_window).median()
        q75 = data[column_name].rolling(window=rolling_window).quantile(0.75)
        q25 = data[column_name].rolling(window=rolling_window).quantile(0.25)
        iqr = q75 - q25
        return (data[column_name] - median) / iqr
    # ----- End Model Formula-----
