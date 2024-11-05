import decimal
from pathlib import Path
import re
from typing import List
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

    def __init__(self, data_sources: List[pd.DataFrame], data_candle: pd.DataFrame, 
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

        # Prepare folder
        coin_name = self.coin if self.coin is not None else ""
        exchange_name = self.exchange if self.exchange is not None else ""
        self.output_folder = os.path.join(output_folder, self.__class__.__name__.lower(), 
                                         coin_name.lower(), exchange_name.lower(), 
                                         self.time_frame.lower(), trading_strategy.name.lower())
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

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
        
        elif self.model == ModelEnum.EMA:
            # Calculating ema for specific column
            # Create upperbound and lowerbound based on ema and diff_threshold
            # Use original column as alpha against threshold
            ema_column_name = f'{column_name}-ema'
            data[ema_column_name] = self.calculate_ema(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[self.lower_threshold_col], data[self.upper_threshold_col] = self.calculate_thresholds(data[ema_column_name], diff_threshold)
            
            return column_name
        elif self.model == ModelEnum.MINMAX:
            # Min-Max Normalization
            minmax_column_name = f'{column_name}-minmax'
            data['rolling_min'] = data[column_name].rolling(rolling_window).min()
            data['rolling_max'] = data[column_name].rolling(rolling_window).max()
            
            # Avoid division by zero by replacing zero differences with a small number (e.g., 1e-8)
            range_diff = data['rolling_max'] - data['rolling_min']
            range_diff = range_diff.replace(0, 1e-8)
            
            # Apply Min-Max normalization to range [-1, 1] using vectorized operations
            data[minmax_column_name] = 2 * (data[column_name] - data['rolling_min']) / range_diff - 1
            
            # Min-Max thresholds typically in range [-1, 1]
            data['lower_threshold'] = -diff_threshold if diff_threshold > 0 else diff_threshold * 2
            data['upper_threshold'] = diff_threshold

            return minmax_column_name
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
    # ----- End Model Formula-----
            
    # ----- Begin Helper -----
    def get_export_file_name(self):
        return self.export_file_name
    
    def validate_data(self, data_source: List[pd.DataFrame], data_candle: pd.DataFrame) -> str:
        for ds in data_source:
            if len(ds) != len(data_candle):
                raise ValueError("One of the data sources and candle have different number of rows")

    def merge_data(self, data_sources: List[pd.DataFrame], data_candle: pd.DataFrame) -> pd.DataFrame:
        self.convert_start_time_column(data_candle)
        merged_data = data_candle.copy()
        
        len_candle = len(data_candle)
        for df in data_sources:
            self.convert_start_time_column(df)
            merged_data = pd.merge(merged_data, df, on='start_time', how='inner')
            
            if len(merged_data) != len_candle:
                print(f"[{self.__class__.__name__}] !!! Warning: One of the data sources and candle have different number of rows. [Candle:{len_candle}, Merged: {len(merged_data)}] !!!")
        
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
