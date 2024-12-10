import decimal
import re
import pandas as pd
import numpy as np

from lib.backtest.backtest_version_enum import BacktestVersionEnum

"""
To run backtest simulation against position column
"""
class BacktestProcessor:
    def __init__(self, 
                 time_frame: str, 
                 rolling_window: int, 
                 diff_threshold: decimal, 
                 trading_strategy_name: str, 
                 trading_fee: decimal, 
                 backtest_version: BacktestVersionEnum=BacktestVersionEnum.V2):
        self.time = self.split_time_frame(time_frame)[0]
        self.frame = self.split_time_frame(time_frame)[1]
        self.rolling_window = rolling_window
        self.diff_threshold = diff_threshold
        self.trading_strategy_name = trading_strategy_name
        self.trading_fee = trading_fee
        self.backtest_version = backtest_version
    
    '''
    Please make sure 'position' column exist before calling this function
    '''
    def run(self, data: pd.DataFrame, rolling_window_start_loc: int):
        # Calculate trade count, close return, daily PnL, cumulative PnL, and drawdown
        data.loc[rolling_window_start_loc:, 'trade'] = abs(data.loc[rolling_window_start_loc:, 'position'].diff())
        
        # --- Calculate PnL ---
        if self.backtest_version == BacktestVersionEnum.V1: 
            data.loc[rolling_window_start_loc:, 'close_return'] = data.loc[rolling_window_start_loc:, 'close'] / data.loc[rolling_window_start_loc:, 'close'].shift(1) - 1
            data.loc[rolling_window_start_loc:, 'daily_PnL'] = (data.loc[rolling_window_start_loc:, 'close_return'] * data.loc[rolling_window_start_loc:, 'position'].shift(1)) \
                                                        - (self.trading_fee * data.loc[rolling_window_start_loc:, 'trade'])
            data.loc[rolling_window_start_loc:, 'cumu_PnL'] = data.loc[rolling_window_start_loc:, 'daily_PnL'].cumsum()
            data.loc[rolling_window_start_loc:, 'drawdown'] = data.loc[rolling_window_start_loc:, 'cumu_PnL'] - data.loc[rolling_window_start_loc:, 'cumu_PnL'].cummax()
        
        elif self.backtest_version == BacktestVersionEnum.V2: 
            data['trade_fee'] = self.trading_fee * data['trade']
            
            # Only for calculating Sharpe ratio
            data.loc[rolling_window_start_loc:, 'close_return'] = data.loc[rolling_window_start_loc:, 'close'] / data.loc[rolling_window_start_loc:, 'close'].shift(1) - 1
            data.loc[rolling_window_start_loc:, 'daily_PnL'] = (data.loc[rolling_window_start_loc:, 'close_return'] * data.loc[rolling_window_start_loc:, 'position'].shift(1)) \
                                                        - data.loc[rolling_window_start_loc:, 'trade_fee']
                                                        
            # Set entry_price only when the position changes and the new position is not 0
            data['entry_price'] = np.nan
            data.loc[(data['position'].diff() != 0) & (data['position'] != 0), 'entry_price'] = data['close']
            
            # Forward fill the entry price for ongoing positions, ensuring it remains NaN if the position is 0
            data['entry_price'] = data['entry_price'].ffill()
            data.loc[data['position'] == 0, 'entry_price'] = np.nan
            data.loc[(data['position'] == 0) & (data['trade'] == 1), 'entry_price'] = data['entry_price'].shift()
            data.loc[:rolling_window_start_loc, 'entry_price'] = np.nan
            
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
            data.loc[rolling_window_start_loc:, 'drawdown'] = data.loc[rolling_window_start_loc:, 'cumu_PnL'] - data.loc[rolling_window_start_loc:, 'cumu_PnL'].cummax()
        
        
        
        # --- Calculate Sharpe ratio and Sortino ratio ---
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
        trade_count = data['trade'].sum()

        # Statistics
        data.loc[0, ''] = np.nan
        data.loc[0, 'Rolling Window'] = self.rolling_window
        data.loc[0, 'Diff Threshold'] = self.diff_threshold
        data.loc[0, 'Trading Strategy'] = self.trading_strategy_name
        data.loc[0, 'Trading Fee'] = self.trading_fee
        data.loc[0, ' '] = np.nan
        data.loc[0, 'Trade Count'] = trade_count
        data.loc[0, 'ADR'] = average_daily_returns
        data.loc[0, 'MDD'] = mdd
        data.loc[0, 'AR'] = annualisedAverageReturn
        data.loc[0, 'CR'] = cumu_pnl
        data.loc[0, 'SR'] = sharpe_ratio
        data.loc[0, 'Sortino'] = sortino_ratio 
        data.loc[0, 'Calmar'] = calmar_ratio 

        return [sharpe_ratio, mdd, cumu_pnl, sortino_ratio, calmar_ratio, trade_count, data]
    
    
    
    # ----- Begin Helper -----
    def get_annual_metric(self):
        if self.frame == "d":
            return 365/self.time
        elif self.frame == "h":
            return (24/self.time) * 365
        elif self.frame == "m":
            return ((60/self.time) * 24) * 365
        else:
            raise ValueError(f"[{self.__class__.__name__}] Invalid timeframe. Please enter 'd', 'h', or 'm'.")
    
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
