import os
import pandas as pd
import numpy as np
from lib.optimization import ModelEnum, Optimization, TradingStrategyEnum

def load_data_sources(parent_dir, coins, time_frames):
    """加载所有数据源"""
    data_sources = {}
    for coin in coins:
        data_sources[coin] = {}
        for time_frame in time_frames:
            data_sources[coin][time_frame] = {
                "coinbase_premium": os.path.join(parent_dir, "resources", f"Cryptoquant_{coin}_MarketCoinbasePremiumIndex_{time_frame}.csv"),
                "candle": os.path.join(parent_dir, "resources", f"binance_candle_{coin}_{time_frame}.csv")
            }
    return data_sources

def calculate_alpha(data_dict):
    """自定义alpha计算逻辑"""
    # 读取数据
    premium_data = pd.read_csv(data_dict["coinbase_premium"])
    
    # 创建结果DataFrame，只保留需要的列
    result = pd.DataFrame({
        "start_time": premium_data["start_time"],
        "alpha": premium_data["coinbase_premium_gap"]  # 直接使用premium gap作为alpha
    })
    
    return result

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
    rolling_windows = list(range(100, 601, 25))  # 100到600，步长25
    diff_thresholds = [round(num, 2) for num in np.arange(0.2, 2.0, 0.2).tolist()]
    trading_fee = 0.00055
    
    # 加载数据源
    data_sources = load_data_sources(parent_dir, coins, time_frames)
    
    # 遍历运行优化
    for coin in coins:
        for time_frame in time_frames:
            for model in models:
                for trading_strategy in trading_strategies:
                    # 读取K线数据
                    data_candle = pd.read_csv(data_sources[coin][time_frame]["candle"])
                    
                    # 计算自定义alpha
                    alpha_data = calculate_alpha(data_sources[coin][time_frame])
                    
                    # 运行优化
                    optimization = Optimization(
                        data_sources=[alpha_data],
                        data_candle=data_candle,
                        rolling_windows=rolling_windows,
                        diff_thresholds=diff_thresholds,
                        trading_strategy=trading_strategy,
                        coin=coin,
                        time_frame=time_frame,
                        model=model,
                        alpha_column_name="alpha",
                        trading_fee=trading_fee,
                        output_folder=f"{parent_dir}/output",
                        export_file_name=f"premium_gap_{model.name}_{coin.upper()}_{time_frame}",
                        is_export_all_chart=True,
                        is_export_all_csv=True
                    )
                    optimization.run()