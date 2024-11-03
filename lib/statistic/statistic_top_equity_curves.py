
import decimal
import gc
import os
from matplotlib import pyplot as plt
import pandas as pd
import heapq

'''
This class is used to generate top n equity curves
'''
class StatisticTopEquityCurves():
    def __init__(self, rolling_windows: list, diff_thresholds: list, top_n: int = 10, initial_capital: float = 10000):
        self.rolling_windows = rolling_windows
        self.diff_thresholds = diff_thresholds
        
        self.top_n = top_n
        self.top_portfolios = []
        self.initial_capital = initial_capital
    
    '''
    Push and keep top n statistics
    '''
    def update_statistic(self, key: str, sharpe_ratio: decimal, data: pd.DataFrame):
        if len(self.top_portfolios) < self.top_n:
            heapq.heappush(self.top_portfolios, (sharpe_ratio, key, data))
        else:
            heapq.heappushpop(self.top_portfolios, (sharpe_ratio, key, data))
    
    '''
    Plots top equity curves
    '''
    def plot_top_equity_curves(self, output_folder: str, export_file_name: str):
        sorted_portfolios = sorted(self.top_portfolios, key=lambda x: x[0], reverse=True)
        
        plt.figure(figsize=(24, 12))
        
        for sharpe_ratio, params, data in sorted_portfolios:
            # Convert cumulative PnL to actual portfolio values
            portfolio_values = self.initial_capital * (1 + data['cumu_PnL'])
            plt.plot(data.index, portfolio_values, label=f"{params}")
        
        # Plot formatting
        plt.title('Top 10 Portfolio Values Over Time (Based on Sharpe Ratio)')
        plt.xlabel('Time')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.legend()
        
        # Format y-axis to show commas
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Save the plot
        portfolio_file_path = os.path.join(output_folder, f"{export_file_name}_top10_portfolios.png")
        plt.tight_layout()
        plt.savefig(portfolio_file_path)
        plt.close()
        gc.collect()
        
        print(f"[{self.__class__.__name__}] Saving top {self.top_n} portfolio chart to '{portfolio_file_path}'")
        