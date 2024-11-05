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
    6. MINMAX                   = ModelEnum.MINMAX  -and-   diff_thresholds between -1 to 1
    '''
    # Parameters
    coins = ["BTC"]
    time_frames = ["1h"]
    models = [ModelEnum.MEAN, ModelEnum.ZSCORE]
    trading_strategies = [TradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM]
    rolling_windows = list(range(100, 601, 25))
    diff_thresholds = [round(num, 2) for num in np.arange(0.2, 2.0, 0.2).tolist()]
    trading_fee = 0.00055 # 0.00055
    
    # Data source
    alpha_column_name = "coinbase_premium_gap"
    alpha_data_sources = {
        coin: {
            time_frame: os.path.join(parent_dir, "resources", f"Cryptoquant_{coin}_MarketCoinbasePremiumIndex_{time_frame}.csv")
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
                    data_source = pd.read_csv(alpha_data_sources[coin][time_frame])
                    data_candle = pd.read_csv(candle_data_source[coin][time_frame])

                    optimization = Optimization(
                        data_sources=[data_source],
                        data_candle=data_candle,
                        rolling_windows=rolling_windows,
                        diff_thresholds=diff_thresholds,
                        trading_strategy=trading_strategy,
                        coin=coin,
                        time_frame=time_frame,
                        model=model,
                        alpha_column_name=alpha_column_name,
                        trading_fee=trading_fee,
                        output_folder=f"{parent_dir}/output",
                        export_file_name=f"{model.name}_{alpha_column_name}_{coin.upper()}_{time_frame}",
                        is_export_all_chart=False,
                        is_export_all_csv=False)
                    optimization.run()
                    
                    