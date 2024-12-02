import decimal
from pathlib import Path
from typing import List, Dict
import pandas as pd
import os
import numpy as np
import threading

from lib.alpha.alpha_config import AlphaConfig
from lib.alpha.alpha_processor import AlphaProcessor
from lib.backtest.backtest_processor import BacktestProcessor
from lib.model.threshold_model_enum import ThresholdModelEnum
from lib.model.threshold_model_processor import ThresholdModelProcessor
from lib.statistic.statistic_chart import StatisticChart
from lib.statistic.statistic_heatmap import StatisticHeatmap
from lib.statistic.statistic_top_equity_curves import StatisticTopEquityCurves
from lib.trading_stategy.threshold_trading_strategy_enum import ThresholdTradingStrategyEnum
from lib.trading_stategy.threshold_trading_strategy_processor import ThresholdTradingStrategyProcessor

"""
This class is designed to simulate trading using rolling windows and differential thresholds, and it automatically generates the SR heatmap.

- data_source and data_candle will be merged automatically based on the 'start_time' column.
"""
# Create a lock for threading
lock = threading.Lock()

class ThresholdOptimization():
    data = pd.DataFrame([])
    rolling_windows = []
    diff_thresholds = []
    trading_strategy = None
    model = None
    export_file_name = ""
    trading_fee = 0.0
    time_frame = ""
    output_folder = ""
    lower_threshold_col = 'lower_threshold'
    upper_threshold_col = 'upper_threshold'
    model_processor = None
    alpha_processor = None
    filter_start_time = None            # To filter data based on start date (E.g: '2024-01-01 12:00:00')
    filter_end_time = None              # To filter data based on end date (E.g: '2024-01-01 12:00:00')

    def __init__(self, data_sources: Dict[str, pd.DataFrame], data_candle: pd.DataFrame, 
                 alpha_config: AlphaConfig,
                 rolling_windows: list, 
                 diff_thresholds: list, 
                 trading_strategy: ThresholdTradingStrategyEnum, 
                 coin: str, 
                 time_frame: str, 
                 model: ThresholdModelEnum,
                 output_folder: str,
                 trading_fee: decimal = 0.00055,
                 exchange: str = None, 
                 enable_alpha_analysis: bool=False,
                 enable_alpha_analysis_confirmation: bool=False,
                 split_heatmap: bool=False,
                 filter_start_time: str=None, filter_end_time: str=None,
                 export_file_name: str="SR_Heatmap"):
        self.rolling_windows = rolling_windows
        self.diff_thresholds = diff_thresholds
        self.trading_strategy = trading_strategy
        self.export_file_name = export_file_name
        self.coin = coin
        self.exchange = exchange
        self.time_frame = time_frame
        self.filter_start_time = filter_start_time
        self.filter_end_time = filter_end_time
        self.model = model
        self.trading_fee = trading_fee
        self.split_heatmap = split_heatmap
        
        # Prepare folder
        coin_name = self.coin if self.coin is not None else ""
        exchange_name = self.exchange if self.exchange is not None else ""
        self.output_folder = os.path.join(output_folder, self.__class__.__name__.lower(), 
                                         coin_name.lower(), exchange_name.lower(), 
                                         self.time_frame.lower(), trading_strategy.name.lower())
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        
        # Merge data
        self.data = self.merge_data(data_sources, data_candle)
        
        # Alpha Processor
        self.alpha_column_name = AlphaProcessor(rolling_windows=rolling_windows,
                                                alpha_config=alpha_config, 
                                                enable_alpha_analysis=enable_alpha_analysis, 
                                                enable_confirmation=enable_alpha_analysis_confirmation, 
                                                output_folder=self.output_folder).run(self.data)
        
        # Trade Processor
        self.model_processor = ThresholdModelProcessor(self.model)
        self.trading_strategy_processor = ThresholdTradingStrategyProcessor(trading_strategy=self.trading_strategy)

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
                    [sharpe_ratio, mdd, cumu_pnl, sortino_ratio, calmar_ratio, data, trade_count] = self.run_backtest(self.data[original_columns], rolling_window, diff_threshold)
                    print(f"[{self.__class__.__name__}] Running [{current_simulation}/{total_simulation}] (RW={rolling_window}, DT={diff_threshold}) SR: {sharpe_ratio}, MDD: {mdd}, cumu_pnl: {cumu_pnl}, Trades: {trade_count}")
                    
                    # Update heatmap statistic
                    self.heatmap.update_statistic(rolling_window, diff_threshold, sharpe_ratio, mdd, cumu_pnl, calmar_ratio, sortino_ratio)

                    # Update top equity curves statistic
                    self.top_equity_curves.update_statistic(f"RW: {rolling_window}, DT: {diff_threshold} (SR: {round(sharpe_ratio, 2)}, MDD: {round(mdd, 2)}, CR: {round(cumu_pnl, 2)})", 
                                                            sharpe_ratio,
                                                            data)
                    
                    if sharpe_ratio > best_sharpe_ratio:
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
        if self.split_heatmap:
            self.heatmap.plot_splitted_2d_heatmap(self.output_folder, self.export_file_name)
        else:
            self.heatmap.plot_2d_heatmap(self.output_folder, self.export_file_name)
        
        # Print best parameters
        self.heatmap._print_best_params(self.coin, self.time_frame, self.model.name)
        
        # Plot top equity curve
        self.top_equity_curves.plot_top_equity_curves(self.output_folder, self.export_file_name)
            
        # Export the best simulation
        if best_data is not None and len(self.export_file_name) > 0:
            # Save signals data for the best parameters
            self._save_best_signals(best_data, best_rolling_window, best_diff_threshold)
            
            file_name = f"{self.export_file_name}_best_{best_rolling_window}_{best_diff_threshold}"
            csv_file_path = os.path.join(self.output_folder, f"{file_name}.csv")
            best_data.to_csv(csv_file_path, index=False)
            print(f"[{self.__class__.__name__}] Saving BEST simulation csv to '{csv_file_path}'")

            chart_file_path = os.path.join(self.output_folder, f"{file_name}.png")
            self.statistic_chart.export_chart(chart_file_path, self.alpha_column_name, best_data)
            print(f"[{self.__class__.__name__}] Saving BEST simulation chart to '{chart_file_path}'")
    
    def _save_best_signals(self, best_data: pd.DataFrame, rolling_window: int, diff_threshold: float):
        """Save the signals data for the best parameters"""
        best_data['signals'] = best_data['position'].diff().fillna(0)
        best_data.loc[best_data['signals'] > 0, 'signals'] = 1    # long buy signal
        best_data.loc[best_data['signals'] < 0, 'signals'] = -1   # short sell signal
        
        signals_data = {
            'rolling_window': rolling_window,
            'diff_threshold': diff_threshold,
            'model': self.model.value,
            'trading_strategy': self.trading_strategy.value,
            'signals': best_data[['signals']].copy()
        }
        
        # Create signals folder
        signals_dir = os.path.join(self.output_folder, 'signals')
        os.makedirs(signals_dir, exist_ok=True)
    
        signals_file = os.path.join(signals_dir, f"{self.export_file_name}_signals.pkl")
        pd.to_pickle(signals_data, signals_file)
        print(f"[{self.__class__.__name__}] Saving best signals data to '{signals_file}'")

    '''
    To run backtest
    '''
    def run_backtest(self, data: pd.DataFrame, rolling_window: int, diff_threshold: decimal):
        # Modeling alpha
        new_alpha_column_name = self.model_processor.run(data, self.alpha_column_name, self.lower_threshold_col, self.upper_threshold_col, rolling_window, diff_threshold)

        # Calculating first valid position
        rolling_window_start_loc = rolling_window - 1
        
        # Apply the trading strategy based on the selected strategy
        data.loc[rolling_window_start_loc:, 'position'] = self.trading_strategy_processor.run(data[rolling_window_start_loc:], new_alpha_column_name, self.lower_threshold_col, self.upper_threshold_col)
        
        # Run backtest
        backtest_results = BacktestProcessor(self.time_frame, rolling_window, diff_threshold, self.trading_strategy.name, self.trading_fee).run(data, rolling_window_start_loc)
        
        # calculate only absolute trade count
        # method 1: count the times changed from 0 to non-0
        position_series = data['position']
        # trade_count = ((position_series != 0) & (position_series.shift(1) == 0)).sum()
        
        # method 2: count the absolute value changes of position, and divide by 2 (because open and close count as one complete trade)
        trade_count = (data['position'].diff() != 0).sum() // 2
        
        return [*backtest_results, trade_count]


    # ----- Begin Helper -----
    def get_export_file_name(self):
        return self.export_file_name
    
    def validate_data(self, data_source: List[pd.DataFrame], data_candle: pd.DataFrame) -> str:
        for ds in data_source:
            if len(ds) != len(data_candle):
                raise ValueError("One of the data sources and candle have different number of rows")

    def merge_data(self, data_sources: Dict[str, pd.DataFrame], data_candle: pd.DataFrame) -> pd.DataFrame:
        """ Merge data sources, ensure time format is unified """
        try:
            # First convert candle data time format
            self.convert_start_time_column(data_candle)
            merged_data = data_candle.copy()[["start_time", "close"]]
            
            # Filter time range
            if (self.filter_start_time is not None):
                filter_start = pd.to_datetime(self.filter_start_time)
                merged_data = merged_data[merged_data['start_time'] >= filter_start]
                
            if (self.filter_end_time is not None):
                filter_end = pd.to_datetime(self.filter_end_time)
                merged_data = merged_data[merged_data['start_time'] <= filter_end]
            
            # Convert and merge each data source
            for alpha_name, df in data_sources.items():
                df_copy = df.copy()  # Create a copy to avoid modifying the original data
                self.convert_start_time_column(df_copy)
                merged_data = pd.merge(merged_data, df_copy, on='start_time', how='inner')
                
                if len(merged_data) != len(data_candle):
                    print(f"[{self.__class__.__name__}] !!! Warning: Alpha {alpha_name} and candle have different number of rows.")
                    print(f"[{self.__class__.__name__}] Candle rows: {len(data_candle)}")
                    print(f"[{self.__class__.__name__}] Merged rows: {len(merged_data)}")
                    print(f"[{self.__class__.__name__}] Please check if there are missing timestamps or mismatched data.")
            
            return merged_data
            
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error merging data: {str(e)}")
            raise e

    def convert_start_time_column(self, df: pd.DataFrame):
        """change time format, support multiple input formats, output as YYYY-MM-DD HH:MM:SS"""
        try:
            # If already datetime format, return directly
            if pd.api.types.is_datetime64_any_dtype(df['start_time']):
                return
            
            # If timestamp format (milliseconds)
            if df['start_time'].dtype == 'int64':
                df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
                return
            
            # Try to parse string format automatically
            try:
                df['start_time'] = pd.to_datetime(df['start_time'])
                return
            except:
                pass
            
            date_formats = [
                '%d-%m-%Y %H:%M:%S',  # DD-MM-YYYY HH:MM:SS
                '%Y-%m-%d %H:%M:%S',  # YYYY-MM-DD HH:MM:SS
                '%m-%d-%Y %H:%M:%S',  # MM-DD-YYYY HH:MM:SS
                '%Y/%m/%d %H:%M:%S',  # YYYY/MM/DD HH:MM:SS
                '%d/%m/%Y %H:%M:%S',  # DD/MM/YYYY HH:MM:SS
                '%m/%d/%Y %H:%M:%S',  # MM/DD/YYYY HH:MM:SS
            ]
            
            for date_format in date_formats:
                try:
                    df['start_time'] = pd.to_datetime(df['start_time'], format=date_format)
                    print(f"[{self.__class__.__name__}] Successfully parsed dates using format: {date_format}")
                    return
                except:
                    continue
            
            # If all formats fail, raise an exception
            raise ValueError(f"[{self.__class__.__name__}] Unable to parse date format. Please ensure dates are in one of the following formats:\n" + 
                            "\n".join(date_formats))
                        
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error converting start_time: {str(e)}")
            print(f"[{self.__class__.__name__}] Sample values from start_time column:")
            print(df['start_time'].head())
            raise e
    # ----- End Helper -----