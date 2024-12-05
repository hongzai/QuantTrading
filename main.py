import os
import pandas as pd
import numpy as np
from lib.alpha.alpha_config import AlphaConfig
from lib.alpha.alpha_combine_method_enum import AlphaCombineMethodEnum
from lib.threshold_optimization import ThresholdModelEnum, ThresholdOptimization, ThresholdTradingStrategyEnum

# Example usage
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    '''
    Use case:
    1. SMA Crossing             = ThresholdModelEnum.MA                 -and-   diff_thresholds with 0
    2. SMA Band                 = ThresholdModelEnum.MA                 -and-   diff_thresholds with any value other than 0
    3. EMA Crossing             = ThresholdModelEnum.EMA                -and-   diff_thresholds with 0
    4. EMA Band                 = ThresholdModelEnum.EMA                -and-   diff_thresholds with any value other than 0
<<<<<<< HEAD
    5. ZSCORE                   = ThresholdModelEnum.ZSCORE             -and-   diff_thresholds
    6. MINMAX                   = ThresholdModelEnum.MINMAX             -and-   diff_thresholds between -1 to 1
    7. Diff From MA             = ThresholdModelEnum.MA_DIFF            -and-   diff_thresholds
    8. Diff From EMA            = ThresholdModelEnum.EMA_DIFF           -and-   diff_thresholds
    9. ROBUST                   = ThresholdModelEnum.ROBUST             -and-   diff_thresholds
    10. RSI                     = ThresholdModelEnum.RSI                -and-   diff_thresholds (lower_threshold=diff_thresholds, upper_threshold=100-diff_threshold) 
    11. Linear regression band  = ThresholdModelEnum.LINEAR_REGRESSION  -and-   diff_thresholds
=======
    5. SMA Reverse              = ThresholdModelEnum.MA_REVERSE         -and-   diff_thresholds with any value other than 0
    6. EMA Reverse              = ThresholdModelEnum.EMA_REVERSE        -and-   diff_thresholds with any value other than 0
    7. Bollinger Band           = ThresholdModelEnum.BOLLINGER          -and-   diff_thresholds with any value other than 0
    8. Bollinger Reverse        = ThresholdModelEnum.BOLLINGER_REVERSE  -and-   diff_thresholds with any value other than 0
    9. ZSCORE                   = ThresholdModelEnum.ZSCORE             -and-   diff_thresholds
    10. MINMAX                   = ThresholdModelEnum.MINMAX             -and-   diff_thresholds between -1 to 1
    11. Diff From MA             = ThresholdModelEnum.MA_DIFF            -and-   diff_thresholds
    12. Diff From EMA            = ThresholdModelEnum.EMA_DIFF           -and-   diff_thresholds
    13. ROBUST                   = ThresholdModelEnum.ROBUST             -and-   diff_thresholds
    14. RSI                     = ThresholdModelEnum.RSI                -and-   diff_thresholds (lower_threshold=diff_thresholds, upper_threshold=100-diff_threshold) 
    15. Linear regression band  = ThresholdModelEnum.LINEAR_REGRESSION  -and-   diff_thresholds
>>>>>>> old-repo/V03
    '''
    # --- 1. Define Parameters ---
    coins = ["BTC"]
    time_frames = ["1h"]
    models = [ThresholdModelEnum.ZSCORE]
<<<<<<< HEAD
    trading_strategies = [ThresholdTradingStrategyEnum.LONG_SHORT_OUTRANGE_MOMEMTUM]
    rolling_windows = list(range(25, 500, 25))
    diff_thresholds = [round(num, 2) for num in np.arange(0.2, 1.5, 0.1).tolist()]
    trading_fee = 0.00055
    enable_alpha_analysis = True                        # To generate data analysis report (Take the first 'rolling_windows' as reference)
=======
    trading_strategies = [ThresholdTradingStrategyEnum.LONG_SHORT_OPPOSITE] 
    rolling_windows = list(range(100, 520, 20))
    diff_thresholds = [round(num, 2) for num in np.arange(0.2, 5.2, 0.2).tolist()]
    trading_fee = 0.00055
    enable_alpha_analysis = False                        # To generate data analysis report (Take the first 'rolling_windows' as reference)
>>>>>>> old-repo/V03
    enable_alpha_analysis_confirmation = False          # To prompt confirmation before starting optimization
    
    # --- 2. Define Data Source ---
    alpha_data_sources = {
        coin: {
            time_frame: {
<<<<<<< HEAD
                "coinbase_premium_gap": os.path.join(parent_dir, "resources", f"Cryptoquant_{coin}_MarketCoinbasePremiumIndex_{time_frame}.csv"),
                "open_interest": os.path.join(parent_dir, "resources", f"Cryptoquant_{coin}_OpenInterest_{time_frame}.csv")
=======
                "optionPC_ratio": os.path.join(parent_dir, "resources", f"Glassnode_aggregated_{coin}_DerivativesOptionsOpenInterestPutCallRatio_{time_frame}.csv")
>>>>>>> old-repo/V03
            }
            for time_frame in time_frames
        }
        for coin in coins
    }
    candle_data_source = {
        coin: {
<<<<<<< HEAD
            time_frame: os.path.join(parent_dir, "resources", f"binance_candle_{coin}_{time_frame}.csv")
=======
            time_frame: os.path.join(parent_dir, "resources", f"bybit_{coin}_linear_{time_frame}_ohlc.csv")
>>>>>>> old-repo/V03
            for time_frame in time_frames
        }
        for coin in coins
    }

    # --- 3. Choosing Alpha ---
    # [Option 1] For 1 alpha
<<<<<<< HEAD
    alpha_config = AlphaConfig.for_1_alpha(alpha_column_name='coinbase_premium_gap')
=======
    alpha_config = AlphaConfig.for_1_alpha(alpha_column_name='v')
>>>>>>> old-repo/V03
    
    # [Option 2] For combine 2 alphas
    #alpha_config = AlphaConfig.for_combine_2_alphas(alpha_column_1='open_interest', alpha_column_2='coinbase_premium_gap', combine_method=AlphaCombineMethodEnum.DIVIDE, weights=[1,1])
    
    # [Option 3] For apply custom formula
    #custom_formula = lambda df: df['open_interest'] / df['coinbase_premium_gap']
    #alpha_config = AlphaConfig.for_custom_formula(custom_formula=custom_formula, new_alpha_column_name="OI-Div-CPG")
    
    
    # --- Iterate optimization process ---
    for coin in coins:
        for time_frame in time_frames:
            for model in models:
                for trading_strategy in trading_strategies:
                    # Read multiple alpha data
                    alpha_dfs = {}
                    for alpha_name, file_path in alpha_data_sources[coin][time_frame].items():
                        alpha_dfs[alpha_name] = pd.read_csv(file_path)
<<<<<<< HEAD

                    # Read candle data
                    data_candle = pd.read_csv(candle_data_source[coin][time_frame])
=======
                        alpha_dfs[alpha_name].rename(columns={'Open Time': 'start_time'}, inplace=True)

                    # Read candle data
                    data_candle = pd.read_csv(candle_data_source[coin][time_frame])
                    data_candle.rename(columns={'timestamp': 'start_time'}, inplace=True)
>>>>>>> old-repo/V03

                    # Create Optimization instance
                    optimization = ThresholdOptimization(
                        data_sources=alpha_dfs,
                        data_candle=data_candle,
                        rolling_windows=rolling_windows,
                        diff_thresholds=diff_thresholds,
                        trading_strategy=trading_strategy,
                        coin=coin,
                        time_frame=time_frame,
                        model=model,
                        alpha_config=alpha_config,
                        enable_alpha_analysis=enable_alpha_analysis,
                        enable_alpha_analysis_confirmation=enable_alpha_analysis_confirmation,
                        trading_fee=trading_fee,
                        #filter_start_time='2023-01-01 12:00:00',
                        #filter_end_time='2024-01-01 12:00:00',
                        split_heatmap=False,
                        output_folder=f"{parent_dir}/output",
                        export_file_name=f"{model.name}_{alpha_config.new_alpha_column_name}_{coin.upper()}_{time_frame}")

                    # Run optimization
                    optimization.run()
                    
                    