import os
import pandas as pd
from lib.optimization import ModelEnum, Optimization, TradingStrategyEnum

# Example usage
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    coin = "BTC"
    time_frame = "1h"
    model = ModelEnum.MEAN
    trading_strategy =TradingStrategyEnum.LONG_ONLY
    alpha_column_name = "coinbase_premium_index"
    
    data_market_coinbase_premium_index = pd.read_csv(f"{parent_dir}/resources/Cryptoquant_{coin}_MarketCoinbasePremiumIndex_{time_frame}.csv")
    data_candle = pd.read_csv(f"{parent_dir}/resources/binance_candle_{coin}_{time_frame}.csv")

    optimization = Optimization(data_sources=[data_market_coinbase_premium_index], 
                data_candle=data_candle,
                rolling_windows=[ 10, 20, 30, 40, 50 ], 
                diff_thresholds=[ 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0 ], 
                trading_strategy=trading_strategy,
                coin=coin,
                model=model,
                time_frame=time_frame,
                alpha_column_name=alpha_column_name,
                output_folder=f"{parent_dir}/output",
                export_file_name=f"coinbase_premium_index_{model.name}_{alpha_column_name}_{coin.upper()}_{time_frame}")
    optimization.run()

                            