import os
import pandas as pd
import numpy as np
from lib.optimization import ModelEnum, Optimization, TradingStrategyEnum
from lib.normalization import Normalization
from lib.normalization_enum import NorModel

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
    models = [ModelEnum.ZSCORE] # ZSCORE,MINMAX,SOFTMAX,ROBUST,MAXABS,LOG,LINEAR_REGRESSION,PERCENTILE,RSI,MEAN,EMA_DIFF,DOUBLE_EMA_CROSSING
    trading_strategies = [TradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM]
    rolling_windows = list(range(5, 205, 5))
    diff_thresholds = [round(num, 2) for num in np.arange(0.2, 3, 0.1).tolist()]
    trading_fee = 0.00055 
    alpha_config = {
        "columns": ['open_interest','coinbase_premium_gap'],
        "method": "divide",  # "add",subtract","multiply","divide","percent_diff","log_ratio","geometric_mean","harmonic_mean"
        "weights": [1, 1]  # 权重配置
    }   
    normalize_models = [NorModel.MINMAX]


    
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
                    # read multiple alpha data
                    alpha_dfs = {}
                    for alpha_name, file_path in alpha_data_sources[coin][time_frame].items():
                        alpha_dfs[alpha_name] = pd.read_csv(file_path)

                    data_candle = pd.read_csv(candle_data_source[coin][time_frame])

                    # 1. First create Optimization instance for alpha config calculation
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

                    # 2. Calculate combined alpha first
                    combined_data = optimization.alpha_preprocessor.combine_alphas(
                        optimization.data,
                        alpha_config["columns"],
                        method=alpha_config["method"]
                    )
                    optimization.data['combined_alpha'] = combined_data

                    # 3. Apply normalization to the combined alpha (if needed)
                    if normalize_models[0] is not None:  # 只有当normalize_model不为None时才进行标准化
                        for normalize_model in normalize_models:
                            normalization = Normalization(
                                model=normalize_model, 
                                rolling_window=20,
                                output_folder=optimization.output_folder
                            )
                            optimization.data['normalized_alpha'] = normalization.normalize(
                                optimization.data, 
                                'combined_alpha'
                            )
                        optimization.alpha_column_name = 'normalized_alpha'  # Use normalized data for trading
                    else:
                        optimization.alpha_column_name = 'combined_alpha'  # Use combined data directly for trading

                    # 4. Run optimization
                    optimization.run()
                    
                    