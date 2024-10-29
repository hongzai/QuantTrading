import os
import pandas as pd
from lib.optimization import ModelEnum, Optimization, TradingStrategyEnum

# Example usage
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Parameter
    coins = [ "BTC" ]
    time_frames = [ "1h" ]
    models = [ ModelEnum.MEAN ]
    trading_strategies = [ TradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM ]
    alpha_column_name = "coinbase_premium_gap"
    rolling_windows = [ 100, 200, 300, 400, 500 ]
    diff_thresholds = [ 0.1, 0.2, 0.3, 0.4, 0.5 ]
    
    # Iterate optimization process
    for coin in coins:
        for time_frame in time_frames:
            for model in models:
                for trading_strategy in trading_strategies:
                    data_market_coinbase_premium_index = pd.read_csv(f"{parent_dir}/resources/Cryptoquant_{coin}_MarketCoinbasePremiumIndex_{time_frame}.csv")
                    data_candle = pd.read_csv(f"{parent_dir}/resources/binance_candle_{coin}_{time_frame}.csv")

                    optimization = Optimization(data_sources=[data_market_coinbase_premium_index], 
                                data_candle=data_candle,
                                rolling_windows=rolling_windows, 
                                diff_thresholds=diff_thresholds, 
                                trading_strategy=trading_strategy,
                                coin=coin,
                                model=model,
                                time_frame=time_frame,
                                alpha_column_name=alpha_column_name,
                                trading_fee= 0.000, # 0.00055
                                output_folder=f"{parent_dir}/output",
                                export_file_name=f"{alpha_column_name}_{model.name}_{alpha_column_name}_{coin.upper()}_{time_frame}",
                                is_export_all_chart=False,
                                is_export_all_csv=False)
                    optimization.run()