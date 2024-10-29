import decimal
from pathlib import Path
from typing import List
import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import threading

from lib.model_enum import ModelEnum
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
        self.model = model
        self.trading_fee = trading_fee
        self.is_export_csv = is_export_all_csv
        self.is_export_chart = is_export_all_chart
        self.alpha_column_name = alpha_column_name

        # Prepare folder
        coin_name = self.coin if self.coin is not None else ""
        exchange_name = self.exchange if self.exchange is not None else ""
        self.output_folder = os.path.join(output_folder, self.__class__.__name__.lower(), coin_name.lower(), exchange_name.lower(), time_frame.lower(), trading_strategy.name.lower())
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        # Merge data
        self.data = self.merge_data(data_sources, data_candle)
        
    def run(self):
        # Logging
        print(f"[{self.__class__.__name__}] Running simulation for ")
        print(f"[{self.__class__.__name__}] - Rolling windows: {self.rolling_windows}")
        print(f"[{self.__class__.__name__}] - Diff Threshold: {self.diff_thresholds}")
        print(f"[{self.__class__.__name__}] - Trading Strategy: {self.trading_strategy.__class__.__name__} : {self.trading_strategy.name}")
        print(f"[{self.__class__.__name__}] - Trading Fee: {self.trading_fee}")

        # Initialize a DataFrame to store the Sharpe ratios
        sharpe_ratios = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        mdds = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        cumu_pnls = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
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
                    [sharpe_ratio, mdd, cumu_pnl, data] = self.calculate_sharpe_ratio(self.data[original_columns], rolling_window, diff_threshold)
                    print(f"[{self.__class__.__name__}] Running [{current_simulation}/{total_simulation}] (RW={rolling_window}, DT={diff_threshold}) SR: {sharpe_ratio}, MDD: {mdd}, cumu_pnl: {cumu_pnl}")
                    
                    sharpe_ratios.loc[rolling_window, diff_threshold] = sharpe_ratio
                    mdds.loc[rolling_window, diff_threshold] = mdd
                    cumu_pnls.loc[rolling_window, diff_threshold] = cumu_pnl

                    if not self.is_export_chart and sharpe_ratio > best_sharpe_ratio:
                        print(f"[{self.__class__.__name__}] Detected BEST simulation for RW={rolling_window} and DT={diff_threshold}")
                        best_sharpe_ratio = sharpe_ratio
                        best_data = data.copy()
                        best_rolling_window = rolling_window
                        best_diff_threshold = diff_threshold
                        
                except Exception as e:
                    sharpe_ratios.loc[rolling_window, diff_threshold] = np.nan
                    print(f"Error for rolling_window={rolling_window} and diff_threshold={diff_threshold}: {e}")

        # Replace NaN values with a default value, e.g., 0
        sharpe_ratios = sharpe_ratios.astype(float).fillna(0)

        # Plot the Sharpe ratio 3D heatmap
        self.plot_2d_heatmap(sharpe_ratios, mdds, cumu_pnls)
        
        # Export the best simulation
        if best_data is not None and len(self.export_file_name) > 0:
            file_name = f"{self.export_file_name}_best_{best_rolling_window}_{best_diff_threshold}"

            csv_file_path = os.path.join(self.output_folder, f"{file_name}.csv")
            best_data.to_csv(csv_file_path, index=False)
            print(f"[{self.__class__.__name__}] Saving BEST simulation csv to '{csv_file_path}'")

            chart_file_path = os.path.join(self.output_folder, f"{file_name}.png")
            self.export_chart(chart_file_path, best_data)
            print(f"[{self.__class__.__name__}] Saving BEST simulation chart to '{chart_file_path}'")
            
    def plot_2d_heatmap(self, sharpe_ratios: pd.DataFrame, mdds: pd.DataFrame, cumu_pnls: pd.DataFrame, plot_sr_only: bool = False) -> str:
        sharpe_ratio_columns = sharpe_ratios.shape[1]

        # Dynamically adjust font size based on the number of columns
        font_size = max(6, min(14, 22 - (sharpe_ratio_columns // 2)))
        
        heatmap_file_path = os.path.join(self.output_folder, f"{self.export_file_name}_SR_Heatmap.png")
        
        # Plot the Sharpe ratio heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(sharpe_ratios, annot=True, annot_kws={"size": font_size}, fmt=".2f", cmap="YlGnBu")
        plt.title("Sharpe Ratio Heatmap")
        plt.xlabel("diff_threshold")
        plt.ylabel("rolling_window")
        plt.savefig(heatmap_file_path)
        plt.close()
        gc.collect()    # Explicit garbage collection

        print(f"[{self.__class__.__name__}] Saving sharpe ratios heatmap to '{heatmap_file_path}'")

        return heatmap_file_path

    # Function to calculate the Sharpe ratio
    def calculate_sharpe_ratio(self, data: pd.DataFrame, rolling_window: int, diff_threshold: decimal):
        self.alpha_handler(data, rolling_window, diff_threshold)

        rolling_window_loc = rolling_window - 1
        
        # Apply the trading strategy based on the selected strategy
        data.loc[rolling_window_loc:, 'position'] = self.trade(data[rolling_window_loc:], self.alpha_column_name)
        
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
        
        # Calculate Sharpe ratio
        annual_metric = self.get_annual_metric()
        average_daily_returns = data['daily_PnL'].mean()
        sharpe_ratio = average_daily_returns / data['daily_PnL'].std() * np.sqrt(annual_metric)
        annualisedAverageReturn = data['daily_PnL'].mean() * annual_metric
        mdd = data['drawdown'].min()
        cumu_pnl = data['cumu_PnL'].iloc[-1]

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

        return [sharpe_ratio, mdd, cumu_pnl, data]


    def export_chart(self, file_path: str, data: pd.DataFrame):
        # --- Process data ---
        columns_to_convert = ['drawdown', 'position', 'close', 'cumu_PnL', self.alpha_column_name, 'SR', 'MDD', 'AR', 'CR', 'Trading Fee', 'Trade Count']
        
        if self.lower_threshold_col in data.columns and self.upper_threshold_col in data.columns:
            columns_to_convert.append(self.lower_threshold_col)
            columns_to_convert.append(self.upper_threshold_col)
            
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # --- Plot the close price and cumu PnL chart ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Add y-axis for price
        ax1.plot(data.index, data['close'], label="Price", color='blue')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Close Price')
        ax1.legend(loc='upper left')

        # Add second y-axis for cumu_PnL
        ax1_2 = ax1.twinx()
        ax1_2.plot(data.index, data['cumu_PnL'], label='Cumu PnL', color='orange')
        ax1_2.set_ylabel('Cumu PnL')
        ax1_2.legend(loc='upper left', bbox_to_anchor=(0, 0.94))

        ax1.set_title(file_path.replace('.png', ''))
        ax1.grid(True)

        # Add additional information as text
        info_text = (
            f"SR: {data['SR'].iloc[0]:.2f}\n"
            f"MDD: {data['MDD'].iloc[0]:.2f}\n"
            f"AR: {data['AR'].iloc[0]:.2f}\n"
            f"CR: {data['CR'].iloc[0]:.2f}\n"
            f"TF: {data['Trading Fee'].iloc[0]}\n"
            f"TC: {data['Trade Count'].iloc[0]}"
        )
        ax1.text(0.01, 0.85, info_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', ha='left', bbox=dict(facecolor='white', alpha=0.5))

        ax1.set_xlabel('Index')
        ax1.set_ylabel('Close Price')
        ax1.legend(loc='upper left')

        # --- Plot the signal and threshold ---
        if self.lower_threshold_col in data.columns and self.upper_threshold_col in data.columns:
            ax2.plot(data.index, data[self.alpha_column_name], label=self.alpha_column_name, color='pink')
            ax2.plot(data.index, data[self.lower_threshold_col], label='Lower Threshold', color='red')
            ax2.plot(data.index, data[self.upper_threshold_col], label='Upper Threshold', color='green')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Value')
            ax2.set_title('Alpha and Thresholds')
            ax2.legend(loc='upper left')
            ax2.grid(True)

            # Focus on the range of lower and upper thresholds
            mid_threshold = (abs(data[self.lower_threshold_col].min()) + abs(data[self.upper_threshold_col].max())) / 2
            ax2.set_ylim([data[self.lower_threshold_col].min() - (mid_threshold * 2.0), data[self.upper_threshold_col].max() + (mid_threshold * 2.0)])

        # --- Plot the drawdown chat ---
        ax3.fill_between(data.index, data['drawdown'], color='red', alpha=0.5, label='Drawdown')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Drawdown')
        ax3.axhline(y=data['drawdown'].mean(), color='red', linestyle='--', label='Drawdown mean')

        # Legends
        ax3.legend(loc='lower left')
        ax3.set_title('Drawdown Over Time')
        ax3.grid(True)

        # --- Display the plot ---
        plt.tight_layout()
        #plt.show()
        plt.savefig(file_path)
        fig.clear()
        plt.close(fig)  # Close the figure to release memory
        gc.collect()    # Explicit garbage collection

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
            mean_column_name = f'{column_name}_mean'
            data[mean_column_name] = self.calculate_mean(data, column_name, rolling_window)
            
            # Use vectorized function to calculate thresholds
            data[self.lower_threshold_col], data[self.upper_threshold_col] = self.calculate_thresholds(data[mean_column_name], diff_threshold)
            
            return column_name
        
        elif self.model == ModelEnum.ZSCORE:
            # Calculating zscore for specific column
            # Create upperbound and lowerbound based on diff_threshold
            # Use zscore as alpha against threshold
            zscore_column_name = f'{column_name}_zscore'
            data[zscore_column_name] = self.calculate_zscore(data, column_name, rolling_window)
            
            data[self.lower_threshold_col] = -diff_threshold if diff_threshold > 0 else diff_threshold * 2
            data[self.upper_threshold_col] = diff_threshold

            return column_name
    # ----- End Model -----

    # ----- Begin trade -----
    def trade(self, data: pd.DataFrame, alpha_column_name: str):
        position = np.zeros(len(data))
        alpha_col = alpha_column_name
        
        if self.trading_strategy == TradingStrategyEnum.LONG_ONLY:
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
        if self.time_frame == "1d":
            return 365
        elif self.time_frame == "1h":
            return 365 * 24
        else:
            raise ValueError(f"[{self.__class__.__name__}] Unknown timeframe. Please configure the metric for the time frame.")
    # ----- End Helper -----