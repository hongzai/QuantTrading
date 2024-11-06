import os
import pandas as pd
import numpy as np
from lib.optimization import ModelEnum, Optimization, TradingStrategyEnum

# Example usage
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    '''
    Use case (Model Template.ipynb):
    1. SMA Crossing             = ModelEnum.MEAN    -and-   diff_thresholds with 0
    2. SMA Band                 = ModelEnum.MEAN    -and-   diff_thresholds with any value greater than 0
    3. EMA Crossing             = ModelEnum.EMA     -and-   diff_thresholds with 0
    4. EMA Band                 = ModelEnum.EMA     -and-   diff_thresholds with any value greater than 0
    5. ZSCORE                   = ModelEnum.ZSCORE  -and-   diff_thresholds with any value greater than 0
    '''
    # Parameters
    coins = ["BTC"]
    time_frames = ["1h"]
    models = [ModelEnum.MEAN, ModelEnum.ZSCORE]
    trading_strategies = [TradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM]
    rolling_windows = list(range(100, 601, 25))
    diff_thresholds = [round(num, 2) for num in np.arange(0.2, 2.0, 0.2).tolist()]
    trading_fee = 0.00055 
    alpha_config = {
        "columns": ['coinbase_premium_gap', 'open_interest'],
        "method": "multiply",  # "add",subtract","multiply","divide","percent_diff","log_ratio","geometric_mean","harmonic_mean"
        "weights": [1, 1]  # 权重配置
    }
    
    # Data source
    alpha_column_name = "coinbase_premium_gap"
    alpha_data_sources = {
        coin: {
            time_frame: {
                "coinbase_premium_gap": os.path.join(parent_dir, "resources", f"Cryptoquant_{coin}_MarketCoinbasePremiumIndex_{time_frame}.csv"),
                "open_interest": os.path.join(parent_dir, "resources", f"Cryptoquant_{coin}_OpenInterest_{time_frame}.csv")
            }
            for time_frame in time_frames
        }
        for coin in coins
    }
    candle_data_source = {
        coin: {
            time_frame: os.path.join(parent_dir, "resources", f"binance_candle_{coin}_{time_frame}.csv")
            for time_frame in time_frames
        }
        for coin in coins
    }

    # Iterate optimization process
    for coin in coins:
        for time_frame in time_frames:
            for model in models:
                for trading_strategy in trading_strategies:
                    # 读取多个alpha数据
                    alpha_dfs = {}
                    for alpha_name, file_path in alpha_data_sources[coin][time_frame].items():
                        alpha_dfs[alpha_name] = pd.read_csv(file_path)

                    data_candle = pd.read_csv(candle_data_source[coin][time_frame])

                    optimization = Optimization(
                        data_sources=alpha_dfs,
                        data_candle=data_candle,
                        alpha_config=alpha_config,
                        rolling_windows=rolling_windows,
                        diff_thresholds=diff_thresholds,
                        trading_strategy=trading_strategy,
                        coin=coin,
                        time_frame=time_frame,
                        model=model,
                        alpha_column_name=alpha_column_name,
                        trading_fee=trading_fee,
                        output_folder=f"{parent_dir}/output",
                        export_file_name=f"{alpha_column_name}_{model.name}_{alpha_column_name}_{coin.upper()}_{time_frame}",
                        is_export_all_chart=False,
                        is_export_all_csv=False)
                    optimization.run()
                    
                    