import os
import pandas as pd
import numpy as np
from lib.optimization import ModelEnum, Optimization, TradingStrategyEnum

# Example usage
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parameters
    coins = ["BTC"]
    times = [1]  # time like: 1, 5, 15, 30, 60
    frames = ["h"]  # input like:d，h，m
    time_frames = [f"{t}{f}" for t in times for f in frames]
    models = [ModelEnum.MEAN, ModelEnum.ZSCORE]
    trading_strategies = [TradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM]
    rolling_windows = list(range(5, 101, 5))
    diff_thresholds = np.arange(0.2, 1.2, 0.2).tolist()
    trading_fee = 0.00055
    
    # Data source
    alpha_column_name = "coinbase_premium_gap"
    alpha_data_sources = {
        time_frame: pd.read_csv(os.path.join(parent_dir, "resources", f"Cryptoquant_BTC_MarketCoinbasePremiumIndex_{time_frame}.csv")) 
        for time_frame in time_frames
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
                    data_source = alpha_data_sources[time_frame]
                    data_candle = pd.read_csv(candle_data_source[coin][time_frame])

                    for time in times:
                        for frame in frames:
                            optimization = Optimization(
                                data_sources=[data_source],
                                data_candle=data_candle,
                                rolling_windows=rolling_windows,
                                diff_thresholds=diff_thresholds,
                                trading_strategy=trading_strategy,
                                coin=coin,
                                time=time,
                                frame=frame,
                                model=model,
                                alpha_column_name=alpha_column_name,
                                trading_fee=trading_fee,
                                output_folder=f"{parent_dir}/output",
                                export_file_name=f"{alpha_column_name}_{model.name}_{alpha_column_name}_{coin.upper()}_{time}{frame}",
                                is_export_all_chart=False,
                                is_export_all_csv=False)
                            optimization.run()
                    
                    