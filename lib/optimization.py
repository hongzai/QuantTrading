import decimal
from pathlib import Path
import re
from typing import List, Dict
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import threading

from lib.model_enum import ModelEnum
from lib.statistic.statistic_chart import StatisticChart
from lib.statistic.statistic_heatmap import StatisticHeatmap
from lib.statistic.statistic_top_equity_curves import StatisticTopEquityCurves
from lib.trading_strategy_enum import TradingStrategyEnum

"""
This class is designed to simulate trading using rolling windows and differential thresholds, and it automatically generates the SR heatmap.

- data_source and data_candle will be merged automatically based on the 'start_time' column.
"""
# Create a lock for threading
lock = threading.Lock()

class AlphaPreprocessor:
    def __init__(self, weights=None, output_folder=None):
        self.preprocessed_column = 'combined_alpha'
        self.weights = weights or [1, 1]  # default weights
        self.output_folder = output_folder
    
    def combine_alphas(self, data: pd.DataFrame, alpha_columns: List[str], method: str = None):
        """
        Full arithmetic operations for combining multiple alpha signals
        
        Parameters:
        - data: DataFrame containing alpha data
        - alpha_columns: List of alpha column names
        - method: ('add', 'subtract', 'multiply', 'divide')
        """
        if len(alpha_columns) == 1:
            return data[alpha_columns[0]]
        alpha1 = data[alpha_columns[0]]
        alpha2 = data[alpha_columns[1]]
        
  
        result = None
        if method == 'add':
            result = alpha1 + alpha2
        elif method == 'subtract':
            result = alpha1 - alpha2
        elif method == 'multiply':
            result = alpha1 * alpha2
        elif method == 'divide':
            # 除法保护
            denominator = alpha2.replace(0, np.nan)  # 将0替换为NaN
            result = alpha1 / denominator
        elif method == 'divide_inverse':
            # 反向除法保护
            denominator = alpha1.replace(0, np.nan)
            result = alpha2 / denominator
        elif method == 'weighted_add':
            # 假设weights在alpha_config中定义
            w1, w2 = self.weights if hasattr(self, 'weights') else (1, 1)
            result = (w1 * alpha1 + w2 * alpha2) / (w1 + w2)
        elif method == 'log_ratio':
            # 对数比率 (处理负值)
            eps = 1e-10  # 小数保护
            result = np.log(np.abs(alpha1) + eps) - np.log(np.abs(alpha2) + eps)
        elif method == 'percent_diff':
            # 百分比差异
            result = (alpha1 - alpha2) / np.abs(alpha2.replace(0, np.nan))
        elif method == 'geometric_mean':
            # 几何平均
            result = np.sqrt(np.abs(alpha1 * alpha2)) * np.sign(alpha1 * alpha2)
        elif method == 'harmonic_mean':
            # 调和平均
            denominator = (1/np.abs(alpha1) + 1/np.abs(alpha2)) / 2
            result = np.sign(alpha1 * alpha2) / denominator
        else:
            raise ValueError(f"Unknown method: {method}. Please choose from: 'add', 'subtract', 'multiply', 'divide', " 
                             f"'divide_inverse', 'weighted_add', 'log_ratio', 'percent_diff', " 
                             f"'geometric_mean', 'harmonic_mean'")

        plt.figure(figsize=(10, 6))
        plt.plot(data.index, alpha1, label=alpha_columns[0])
        plt.plot(data.index, alpha2, label=alpha_columns[1])
        plt.plot(data.index, result, label=f'Combined ({method})')
        plt.title(f'{alpha_columns[0]} {method} {alpha_columns[1]}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        output_path = os.path.join(self.output_folder, f'combined_alpha_{alpha_columns[0]}_{method}_{alpha_columns[1]}.png')
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        

        print(f"\nCombined Alpha Chart: {output_path}")
        user_input = input("\nContinue? (y/n): ").lower()
        if user_input != 'y':
            print("User choose to terminate the program")
            import sys
            sys.exit(0)

        return result

class Optimization():

    data = pd.DataFrame([])
    rolling_windows = []
    diff_thresholds = []
    trading_strategy = None
    model = None
    export_file_name = ""
    trading_fee = 0.0
    time_frame = ""
    output_folder = ""
    is_export_csv = None
    is_export_chart = None
    alpha_column_name = ""
    lower_threshold_col = 'lower_threshold'
    upper_threshold_col = 'upper_threshold'
    alpha_preprocessor = AlphaPreprocessor()
    alpha_columns = ['coinbase_premium_gap', 'open_interest']

    def __init__(self, data_sources: Dict[str, pd.DataFrame], data_candle: pd.DataFrame, 
                 alpha_config: dict,
                 alpha_column_name: str, rolling_windows: list, diff_thresholds: list, 
                 trading_strategy: TradingStrategyEnum, coin: str, time_frame: str, model: ModelEnum,
                 output_folder: str,
                 trading_fee: decimal = 0.00055,
                 exchange: str = None, 
                 export_file_name: str="SR_Heatmap", is_export_all_csv: bool=True, is_export_all_chart: bool=True):
        self.rolling_windows = rolling_windows
        self.diff_thresholds = diff_thresholds
        self.trading_strategy = trading_strategy
        self.export_file_name = export_file_name
        self.coin = coin
        self.exchange = exchange
        self.time_frame = time_frame
        self.time = self.split_time_frame(time_frame)[0]
        self.frame = self.split_time_frame(time_frame)[1]
        self.model = model
        self.trading_fee = trading_fee
        self.is_export_csv = is_export_all_csv
        self.is_export_chart = is_export_all_chart
        self.alpha_column_name = alpha_column_name
        self.alpha_columns = alpha_config["columns"]
        self.alpha_method = alpha_config.get("method", None)
        self.alpha_weights = alpha_config.get("weights", None)
        
        # Prepare folder
        coin_name = self.coin if self.coin is not None else ""
        exchange_name = self.exchange if self.exchange is not None else ""
        self.output_folder = os.path.join(output_folder, self.__class__.__name__.lower(), 
                                         coin_name.lower(), exchange_name.lower(), 
                                         self.time_frame.lower(), trading_strategy.name.lower())
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        self.alpha_preprocessor = AlphaPreprocessor(
            weights=self.alpha_weights if len(self.alpha_columns) > 1 else None,
            output_folder=self.output_folder
        )

        # Merge data
        self.data = self.merge_data(data_sources, data_candle)
        
        # Statistic
        self.heatmap = StatisticHeatmap(rolling_windows, diff_thresholds)
        self.top_equity_curves = StatisticTopEquityCurves(rolling_windows, diff_thresholds)
        self.statistic_chart = StatisticChart(rolling_windows, diff_thresholds)
        
    def run(self):
        # Logging
        print(f"[{self.__class__.__name__}] Running simulation for ")
        print(f"[{self.__class__.__name__}] - Rolling windows: {self.rolling_windows}")
        print(f"[{self.__class__.__name__}] - Diff Threshold: {self.diff_thresholds}")
        print(f"[{self.__class__.__name__}] - Trading Strategy: {self.trading_strategy.__class__.__name__} : {self.trading_strategy.name}")
        print(f"[{self.__class__.__name__}] - Trading Fee: {self.trading_fee}")

        # Initialize DataFrames
        original_columns = self.data.columns
        total_simulation = len(self.rolling_windows) * len(self.diff_thresholds)
        current_simulation = 0
        
        best_sharpe_ratio = -np.inf
        best_data = None
        best_rolling_window = None
        best_diff_threshold = None
        
        # Iterate through each combination of rolling_window and diff_threshold
        for rolling_window in self.rolling_windows:
            for diff_threshold in self.diff_thresholds:
                try:
                    # Starting simulation
                    current_simulation+=1

                    # Running simulation
                    [sharpe_ratio, mdd, cumu_pnl, sortino_ratio, calmar_ratio, data] = self.calculate_sharpe_ratio(self.data[original_columns], rolling_window, diff_threshold)
                    print(f"[{self.__class__.__name__}] Running [{current_simulation}/{total_simulation}] (RW={rolling_window}, DT={diff_threshold}) SR: {sharpe_ratio}, MDD: {mdd}, cumu_pnl: {cumu_pnl}")
                    
                    # Update heatmap statistic
                    self.heatmap.update_statistic(rolling_window, diff_threshold, sharpe_ratio, mdd, cumu_pnl, calmar_ratio, sortino_ratio)

                    # Update top equity curves statistic
                    self.top_equity_curves.update_statistic(f"RW: {rolling_window}, DT: {diff_threshold} (SR: {round(sharpe_ratio, 2)}, MDD: {round(mdd, 2)}, CR: {round(cumu_pnl, 2)})", 
                                                            sharpe_ratio,
                                                            data)
                    
                    if not self.is_export_chart and sharpe_ratio > best_sharpe_ratio:
                        print(f"[{self.__class__.__name__}] Detected BEST simulation for RW={rolling_window} and DT={diff_threshold}")
                        best_sharpe_ratio = sharpe_ratio
                        best_data = data.copy()
                        best_rolling_window = rolling_window
                        best_diff_threshold = diff_threshold
                        
                except Exception as e:
                    self.heatmap.update_statistic(rolling_window, diff_threshold, np.nan, np.nan, np.nan, np.nan, np.nan)
                    print(f"Error for rolling_window={rolling_window} and diff_threshold={diff_threshold}: {e}")

        # Plot heatmap
        self.heatmap.fill_nan_statistic()
        self.heatmap.plot_2d_heatmap(self.output_folder, self.export_file_name)
        
        # Print best parameters
        self.heatmap._print_best_params(self.coin, self.time_frame, self.model.name)
        
        # Plot top equity curve
        self.top_equity_curves.plot_top_equity_curves(self.output_folder, self.export_file_name)
            
        # Export the best simulation
        if best_data is not None and len(self.export_file_name) > 0:
            file_name = f"{self.export_file_name}_best_{best_rolling_window}_{best_diff_threshold}"

            csv_file_path = os.path.join(self.output_folder, f"{file_name}.csv")
            best_data.to_csv(csv_file_path, index=False)
            print(f"[{self.__class__.__name__}] Saving BEST simulation csv to '{csv_file_path}'")

            chart_file_path = os.path.join(self.output_folder, f"{file_name}.png")
            self.statistic_chart.export_chart(chart_file_path, self.alpha_column_name, best_data)
            print(f"[{self.__class__.__name__}] Saving BEST simulation chart to '{chart_file_path}'")
            

    # Function to calculate the Sharpe ratio
    def calculate_sharpe_ratio(self, data: pd.DataFrame, rolling_window: int, diff_threshold: decimal):
        new_alpha_column_name = self.alpha_handler(data, rolling_window, diff_threshold)

        rolling_window_loc = rolling_window - 1
        
        # Apply the trading strategy based on the selected strategy
        data.loc[rolling_window_loc:, 'position'] = self.trade(data[rolling_window_loc:], new_alpha_column_name)
        
        # Calculate trade count, close return, daily PnL, cumulative PnL, and drawdown
        data.loc[rolling_window_loc:, 'trade'] = abs(data.loc[rolling_window_loc:, 'position'].diff())
        
        '''
        # ----- Version 1 -----
        data.loc[rolling_window_loc:, 'close_return'] = data.loc[rolling_window_loc:, 'close'] / data.loc[rolling_window_loc:, 'close'].shift(1) - 1
        data.loc[rolling_window_loc:, 'daily_PnL'] = (data.loc[rolling_window_loc:, 'close_return'] * data.loc[rolling_window_loc:, 'position'].shift(1)) \
                                                      - (self.trading_fee * data.loc[rolling_window_loc:, 'trade'])
        data.loc[rolling_window_loc:, 'cumu_PnL'] = data.loc[rolling_window_loc:, 'daily_PnL'].cumsum()
        data.loc[rolling_window_loc:, 'drawdown'] = data.loc[rolling_window_loc:, 'cumu_PnL'] - data.loc[rolling_window_loc:, 'cumu_PnL'].cummax()
        '''
        
        # ----- Version 2 -----
        data['trade_fee'] = self.trading_fee * data['trade']
        
        # Only for calculating Sharpe ratio
        data.loc[rolling_window_loc:, 'close_return'] = data.loc[rolling_window_loc:, 'close'] / data.loc[rolling_window_loc:, 'close'].shift(1) - 1
        data.loc[rolling_window_loc:, 'daily_PnL'] = (data.loc[rolling_window_loc:, 'close_return'] * data.loc[rolling_window_loc:, 'position'].shift(1)) \
                                                      - data.loc[rolling_window_loc:, 'trade_fee']
                                                      
        # Set entry_price only when the position changes and the new position is not 0
        data['entry_price'] = np.nan
        data.loc[(data['position'].diff() != 0) & (data['position'] != 0), 'entry_price'] = data['close']
        
        # Forward fill the entry price for ongoing positions, ensuring it remains NaN if the position is 0
        data['entry_price'] = data['entry_price'].ffill()
        data.loc[data['position'] == 0, 'entry_price'] = np.nan
        data.loc[(data['position'] == 0) & (data['trade'] == 1), 'entry_price'] = data['entry_price'].shift()
        data.loc[:rolling_window_loc, 'entry_price'] = np.nan
        
        # Calculate daily PnL as a percentage
        data['unrealized_PnL'] = np.where(
            (data['position'] != 0),
            ((data['close'] - data['entry_price']) / data['entry_price'] * data['position'].shift(1)),
            0
        )
        
        # Calculate realized PnL when the position is closed
        data['realized_PnL'] = np.where(
            (data['position'].shift(1) != 0) & (data['position'] != data['position'].shift(1)),
            ((data['close'] - data['entry_price'].shift(1)) / data['entry_price'].shift(1) * data['position'].shift(1) - (data['trade_fee'])),
            0.0 - data['trade_fee']
        )
        data['realized_PnL'] = data['realized_PnL'].cumsum()

        # Calculate cumulative PnL as the sum of realized and unrealized PnL
        data['cumu_PnL'] = data['realized_PnL'] + data['unrealized_PnL']
        
        # Calculate drawdown as the percentage difference from the running maximum
        data.loc[rolling_window_loc:, 'drawdown'] = data.loc[rolling_window_loc:, 'cumu_PnL'] - data.loc[rolling_window_loc:, 'cumu_PnL'].cummax()
        
        # Calculate Sharpe ratio and Sortino ratio
        annual_metric = self.get_annual_metric()
        average_daily_returns = data['daily_PnL'].mean()
        
        # Sharpe ratio calculation
        sharpe_ratio = average_daily_returns / data['daily_PnL'].std() * np.sqrt(annual_metric)
        
        # Sortino ratio calculation
        excess_returns = data['daily_PnL'] - 0  # 0 is the minimum acceptable return (MAR), Can be adjusted as needed
        negative_excess = excess_returns[excess_returns < 0]

        if len(negative_excess) > 0:
            downside_deviation = np.sqrt(np.mean(negative_excess ** 2))
            sortino_ratio = (average_daily_returns / downside_deviation) * np.sqrt(annual_metric)
        else:
            sortino_ratio = np.inf
    
        # Calmar ratio calculation
        annualisedAverageReturn = average_daily_returns * annual_metric
        mdd = data['drawdown'].min()
        cumu_pnl = data['cumu_PnL'].iloc[-1]
        calmar_ratio = abs(annualisedAverageReturn / mdd) if mdd != 0 else np.inf

        # Statistics
        data.loc[0, ''] = np.nan
        data.loc[0, 'Rolling Window'] = rolling_window
        data.loc[0, 'Diff Threshold'] = diff_threshold
        data.loc[0, 'Trading Strategy'] = self.trading_strategy.name
        data.loc[0, 'Trading Fee'] = self.trading_fee
        data.loc[0, ' '] = np.nan
        data.loc[0, 'Trade Count'] = data['trade'].sum()
        data.loc[0, 'ADR'] = average_daily_returns
        data.loc[0, 'MDD'] = mdd
        data.loc[0, 'AR'] = annualisedAverageReturn
        data.loc[0, 'CR'] = cumu_pnl
        data.loc[0, 'SR'] = sharpe_ratio
        data.loc[0, 'Sortino'] = sortino_ratio 
        data.loc[0, 'Calmar'] = calmar_ratio 

        # Export simulation results and chart to a CSV file
        if len(self.export_file_name) > 0:
            file_name = f"{self.export_file_name}_{rolling_window}_{diff_threshold}"

            if self.is_export_csv:
                file_path = os.path.join(self.output_folder, f"{file_name}.csv")
                data.to_csv(file_path, index=False)
                print(f"[{self.__class__.__name__}] Saving simulation csv to '{file_path}'")

            if self.is_export_chart:
                file_path = os.path.join(self.output_folder, f"{file_name}.png")
                self.export_chart(file_path, data)
                print(f"[{self.__class__.__name__}] Saving simulation chart to '{file_path}'")

        return [sharpe_ratio, mdd, cumu_pnl, sortino_ratio, calmar_ratio, data]


    # ----- Begin Model -----
    '''
    To provide some standard alpha models for alpha handler easily.
    '''
    def alpha_handler(self, data: pd.DataFrame, rolling_window: int, diff_threshold: decimal):
        column_name = self.alpha_column_name
        
        if self.model == ModelEnum.MEAN:
            # Calculate sma for specific column
            # Create upperbound and lowerbound based on sma and diff_threshold
            # Use original column as alpha against threshold
            mean_column_name = f'{column_name}-mean'
            data[mean_column_name] = self.calculate_mean(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[self.lower_threshold_col], data[self.upper_threshold_col] = self.calculate_thresholds(data[mean_column_name], diff_threshold)
            
            return column_name
        
        elif self.model == ModelEnum.ZSCORE:
            # Calculating zscore for specific column
            # Create upperbound and lowerbound based on diff_threshold
            # Use zscore as alpha against threshold
            zscore_column_name = f'{column_name}-zscore'
            data[zscore_column_name] = self.calculate_zscore(data, column_name, rolling_window)
            
            data[self.lower_threshold_col] = -diff_threshold if diff_threshold > 0 else diff_threshold * 2
            data[self.upper_threshold_col] = diff_threshold

            return zscore_column_name
        
        elif self.model == ModelEnum.EMA_DIFF:
            # Calculating ema for specific column
            ema_column_name = f'{column_name}-ema'
            data[ema_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            # difference between alpha and ema
            data['alpha_ema_diff'] = (data[column_name] - data[ema_column_name]) / data[ema_column_name]
            
            # set thresholds as percentage of difference
            data[self.lower_threshold_col] = -diff_threshold
            data[self.upper_threshold_col] = diff_threshold
            
            return 'alpha_ema_diff'

        elif self.model == ModelEnum.MINMAX:
            # Min-Max Scaling with rolling window
            minmax_column_name = f'{column_name}-minmax'

            rolling_min = data[column_name].rolling(window=rolling_window).min()
            rolling_max = data[column_name].rolling(window=rolling_window).max()

            # Calculate normalized values
            data[minmax_column_name] = (data[column_name] - rolling_min) / (rolling_max - rolling_min)
            

            data[self.lower_threshold_col] = diff_threshold
            data[self.upper_threshold_col] = 1 - diff_threshold
            
            return minmax_column_name

        elif self.model == ModelEnum.ROBUST:
            # Robust Scaling with rolling window
            robust_column_name = f'{column_name}-robust'
            
            # Use rolling window to calculate median and IQR
            rolling_median = data[column_name].rolling(window=rolling_window).median()
            q75 = data[column_name].rolling(window=rolling_window).quantile(0.75)
            q25 = data[column_name].rolling(window=rolling_window).quantile(0.25)
            rolling_iqr = q75 - q25
            

            data[robust_column_name] = (data[column_name] - rolling_median) / rolling_iqr
            

            data[self.lower_threshold_col] = -diff_threshold
            data[self.upper_threshold_col] = diff_threshold
            
            return robust_column_name

        elif self.model == ModelEnum.MAXABS:
            # MaxAbs Scaling with rolling window
            maxabs_column_name = f'{column_name}-maxabs'
            
            rolling_maxabs = data[column_name].abs().rolling(window=rolling_window).max()
            
            # Calculate maxabs scaling
            data[maxabs_column_name] = data[column_name] / rolling_maxabs
            
            data[self.lower_threshold_col] = -diff_threshold
            data[self.upper_threshold_col] = diff_threshold
            
            return maxabs_column_name

        elif self.model == ModelEnum.LOG:
            # Log Transformation with thresholds
            log_column_name = f'{column_name}-log'
            eps = 1e-10  # avoid log = 0
            
            data[log_column_name] = np.log(np.abs(data[column_name]) + eps)
            
            # Use rolling window to calculate thresholds
            rolling_mean = data[log_column_name].rolling(window=rolling_window).mean()
            rolling_std = data[log_column_name].rolling(window=rolling_window).std()
            
 
            data[self.lower_threshold_col] = rolling_mean - (diff_threshold * rolling_std)
            data[self.upper_threshold_col] = rolling_mean + (diff_threshold * rolling_std)
            
            return log_column_name

        elif self.model == ModelEnum.SOFTMAX:
            # SoftMax with rolling window
            softmax_column_name = f'{column_name}-softmax'
            
            # Use numpy's vectorized operation
            values = data[column_name].values
            result = np.zeros(len(data))
            
            for i in range(rolling_window - 1, len(data)):
                window = values[i-rolling_window+1:i+1]
                # Numerical stability processing
                window = window - np.max(window)
                exp_window = np.exp(window)
                softmax = exp_window / np.sum(exp_window)
                result[i] = softmax[-1]
            
            data[softmax_column_name] = result
            # use bfill to replace fillna(method='bfill')
            data[softmax_column_name] = data[softmax_column_name].bfill()
            
            
            data[self.lower_threshold_col] = diff_threshold
            data[self.upper_threshold_col] = 1 - diff_threshold
            
            print(f"[{self.__class__.__name__}] Softmax values range: {data[softmax_column_name].min():.4f} to {data[softmax_column_name].max():.4f}")
            
            return softmax_column_name

        elif self.model == ModelEnum.DOUBLE_EMA_CROSSING:
            # Calculate long and short EMA
            long_ema_name = f'{column_name}-long-ema'
            short_ema_name = f'{column_name}-short-ema'
            
            # long_window = rolling_window
            # short_window = diff_threshold (as integer)
            short_window = int(diff_threshold)
            
            # Calculate both EMAs
            data[long_ema_name] = self.calculate_ema(data, column_name, rolling_window)
            data[short_ema_name] = self.calculate_ema(data, column_name, short_window)
            
            # Calculate the difference between short and long EMA
            data['ema_cross'] = data[short_ema_name] - data[long_ema_name]
            data['prev_ema_cross'] = data['ema_cross'].shift(1)
            
            # Set thresholds for crossing signals
            data[self.lower_threshold_col] = np.where(
                (data['ema_cross'] < 0) & (data['prev_ema_cross'] >= 0),  
                -1,  
                0 
            )
            
            data[self.upper_threshold_col] = np.where(
                (data['ema_cross'] > 0) & (data['prev_ema_cross'] <= 0),  
                1,  
                0 
            )
            
            print(f"[{self.__class__.__name__}] Double EMA Crossing (Long={rolling_window}, Short={short_window})")
            
            return 'ema_cross'

        elif self.model == ModelEnum.RSI:
            # Calculate RSI using the rolling window as period
            rsi_column_name = f'{column_name}-rsi'
            
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
            data[rsi_column_name] = 100 - (100 / (1 + rs))
            
            data[self.lower_threshold_col] = diff_threshold
            data[self.upper_threshold_col] = 100 - diff_threshold
            
            print(f"[{self.__class__.__name__}] RSI Period={rolling_window}, Thresholds={data[self.lower_threshold_col].iloc[-1]}/{data[self.upper_threshold_col].iloc[-1]}")
            
            return rsi_column_name

        elif self.model == ModelEnum.LINEAR_REGRESSION:
            # 线性回归
            regression_column_name = f'{column_name}-regression'
            data[regression_column_name] = self.calculate_linear_regression(data, column_name, rolling_window)
            
            data[self.lower_threshold_col] = data[regression_column_name] - diff_threshold
            data[self.upper_threshold_col] = data[regression_column_name] + diff_threshold
            
            return regression_column_name

        elif self.model == ModelEnum.PERCENTILE:
            # 百分位数
            percentile_column_name = f'{column_name}-percentile'
            data[percentile_column_name] = self.calculate_percentile(data, column_name, rolling_window)
            
            data[self.lower_threshold_col] = data[percentile_column_name].quantile(0.25)
            data[self.upper_threshold_col] = data[percentile_column_name].quantile(0.75)
            
            return percentile_column_name
    # ----- End Model -----

    # ----- Begin trade -----
    def trade(self, data: pd.DataFrame, alpha_column_name: str):
        position = np.zeros(len(data))
        alpha_col = alpha_column_name
        
        if self.trading_strategy == TradingStrategyEnum.LONG_ABOVE_UPPER:
            position[data[alpha_col] > data[self.upper_threshold_col]] = 1
        elif self.trading_strategy == TradingStrategyEnum.SHORT_BELOW_LOWER:
            position[data[alpha_col] < data[self.lower_threshold_col]] = -1
        if self.trading_strategy == TradingStrategyEnum.LONG_ABOVE_UPPER:
            position[data[alpha_col] > data[self.upper_threshold_col]] = 1
        elif self.trading_strategy == TradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM:
            position[data[alpha_col] > data[self.upper_threshold_col]] = 1
            position[data[alpha_col] < data[self.lower_threshold_col]] = -1
        elif self.trading_strategy == TradingStrategyEnum.LONG_SHORT_INRANGE_MEAN_REVERSION:
            position[data[alpha_col] > data[self.upper_threshold_col]] = -1
            position[data[alpha_col] < data[self.lower_threshold_col]] = 1
        elif self.trading_strategy == TradingStrategyEnum.LONG_SHORT_OPPOSITE:
            for i in range(0, len(data)):
                previous_position = position[i-1]
                alpha_value = data[alpha_col].iloc[i]
                lower_threshold = data[self.lower_threshold_col].iloc[i]
                upper_threshold = data[self.upper_threshold_col].iloc[i]
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
    
    def calculate_mean(self, data: pd.DataFrame, column_name: str, rolling_window: decimal):
        return data[column_name].rolling(rolling_window).mean()
    
    def calculate_ema(self, data, column_name, span):
        return data[column_name].ewm(span=span, adjust=False).mean()

    # 添加计算线性回归的方法
    def calculate_linear_regression(self, data: pd.DataFrame, column_name: str, rolling_window: int):
        # 使用rolling apply来计算线性回归
        def linear_regression(x):
            x = np.arange(len(x))
            y = data[column_name].iloc[x.index]
            slope, intercept = np.polyfit(x, y, 1)
            return slope * x[-1] + intercept
        
        return data[column_name].rolling(window=rolling_window).apply(linear_regression, raw=False)

    # 添加计算百分位数的方法
    def calculate_percentile(self, data: pd.DataFrame, column_name: str, rolling_window: int):
        return data[column_name].rolling(window=rolling_window).apply(lambda x: np.percentile(x, 50), raw=False)
    # ----- End Model Formula-----
            
    # ----- Begin Helper -----
    def get_export_file_name(self):
        return self.export_file_name
    
    def validate_data(self, data_source: List[pd.DataFrame], data_candle: pd.DataFrame) -> str:
        for ds in data_source:
            if len(ds) != len(data_candle):
                raise ValueError("One of the data sources and candle have different number of rows")

    def merge_data(self, data_sources: Dict[str, pd.DataFrame], data_candle: pd.DataFrame) -> pd.DataFrame:
        self.convert_start_time_column(data_candle)
        merged_data = data_candle.copy()
        
        # loop alpha data sources and merge
        for alpha_name, df in data_sources.items():
            self.convert_start_time_column(df)
            merged_data = pd.merge(merged_data, df, on='start_time', how='inner')
            
            if len(merged_data) != len(data_candle):
                print(f"[{self.__class__.__name__}] !!! Warning: Alpha {alpha_name} and candle have different number of rows. [Candle:{len(data_candle)}, Merged: {len(merged_data)}] !!!")
        
        # 只有多个因子时才需要计算combined alpha
        if len(self.alpha_columns) > 1 and self.alpha_method:
            merged_data['combined_alpha'] = self.alpha_preprocessor.combine_alphas(
                merged_data, 
                self.alpha_columns,
                method=self.alpha_method
            )
        
        return merged_data

    def convert_start_time_column(self, df: pd.DataFrame):
        # Convert 'start_time' to datetime
        if df['start_time'].dtype == 'int64':
            df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
        else:
            df['start_time'] = pd.to_datetime(df['start_time'])

    def get_annual_metric(self):
        if self.frame == "d":
            return 365/self.time
        elif self.frame == "h":
            return (24/self.time) * 365
        elif self.frame == "m":
            return ((60/self.time) * 24) * 365
        else:
            raise ValueError(f"[{self.__class__.__name__}] Invalid timeframe. Please enter 'd', 'h', or 'm'.")

    def calculate_mdd(self, cumulative_returns):
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        mdd = drawdown.min()
        return abs(mdd)
    
    def split_time_frame(self, time_frame: str):
        # Use regular expression to separate numeric and alphabetic parts
        match = re.match(r"(\d+)([a-zA-Z]+)", time_frame)
        if match:
            time = int(match.group(1))  # Convert to integer if needed
            frame = match.group(2)
            
            return [time, frame]
        else:
            raise ValueError(f"[{self.__class__.__name__}] Invalid timeframe. Please enter correct format. E.g: '1d', '1h', or '1m'.")
    # ----- End Helper -----