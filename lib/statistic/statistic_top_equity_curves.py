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
    def __init__(self, rolling_windows: list, diff_thresholds: list, top_n: int = 15, initial_capital: float = 10000):
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
        
        # create a chart with two subplots: the top is the equity curves, the bottom is the table
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 16), 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # draw the equity curves
        for sharpe_ratio, params, data in sorted_portfolios:
            portfolio_values = self.initial_capital * (1 + data['cumu_PnL'])
            ax1.plot(data.index, portfolio_values, label=f"{params}")
        
    
        ax1.set_title(f'Top {self.top_n} Portfolio Values Over Time (Based on Sharpe Ratio)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # create the table data
        table_data = []
        for sharpe_ratio, params, data in sorted_portfolios:
            trade_count = (data['position'].diff() != 0).sum() // 2
            mdd = data['drawdown'].min()
            cr = data['cumu_PnL'].iloc[-1]
            
            # add all metrics to the table row
            table_data.append([
                params,                    # Parameters
                f"{trade_count:,}",        # Trade Count
                f"{sharpe_ratio:.2f}",     # Sharpe Ratio
                f"{mdd:.2%}",              # Maximum Drawdown
                f"{cr:.2%}"                # Cumulative Return
            ])
        
        # create the table
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=table_data,
                         colLabels=['Parameters', 'Trade Count', 'SR', 'MDD', 'CR'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
        
        # set the table style
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)
        
        # set the table title
        ax2.set_title('Performance Metrics for Each Portfolio', pad=20)
        

        portfolio_file_path = os.path.join(output_folder, f"{export_file_name}_top{self.top_n}_portfolios.png")
        plt.tight_layout()
        plt.savefig(portfolio_file_path, bbox_inches='tight', dpi=300)
        plt.close()
        gc.collect()
        
        print(f"[{self.__class__.__name__}] Saving top {self.top_n} portfolio chart to '{portfolio_file_path}'")
        