
import decimal
import gc
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

'''
This class is used to generate optimization heatmap
'''
class StatisticHeatmap():
    def __init__(self, rolling_windows: list, diff_thresholds: list):
        self.rolling_windows = rolling_windows
        self.diff_thresholds = diff_thresholds
    
        self.sharpe_ratios = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        self.mdds = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        self.cumu_pnls = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        self.calmar_ratios = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        self.sortino_ratios = pd.DataFrame(index=self.rolling_windows, columns=self.diff_thresholds)
        
    '''
    Updates the calculated statistic for the provided rolling window and difference threshold.    
    '''
    def update_statistic(self, rolling_window: int, diff_threshold: decimal, sharpe_ratio: decimal, mdd: decimal, cumu_pnl: decimal, calmar_ratio: decimal, sortino_ratio: decimal):
        self.sharpe_ratios.loc[rolling_window, diff_threshold] = sharpe_ratio
        self.mdds.loc[rolling_window, diff_threshold] = mdd
        self.cumu_pnls.loc[rolling_window, diff_threshold] = cumu_pnl
        self.calmar_ratios.loc[rolling_window, diff_threshold] = calmar_ratio
        self.sortino_ratios.loc[rolling_window, diff_threshold] = sortino_ratio
    
    '''
    Fills any NaN values in the dataframes with a default value (0)
    '''
    def fill_nan_statistic(self):
        self.sharpe_ratios = self.sharpe_ratios.astype(float).fillna(0)
        self.mdds = self.mdds.astype(float).fillna(0)
        self.cumu_pnls = self.cumu_pnls.astype(float).fillna(0)
        self.calmar_ratios = self.calmar_ratios.astype(float).fillna(0)
        self.sortino_ratios = self.sortino_ratios.astype(float).fillna(0)  
        
    '''
    Plots 2D heatmaps for each statistic (Sharpe Ratio, MDD, Cumulative PnL, Calmar Ratio, Sortino Ratio) or only the Sharpe ratio if specified.
    '''
    def plot_2d_heatmap(self, output_folder: str, export_file_name: str) -> str:
        sharpe_ratio_columns = self.sharpe_ratios.shape[1]

        # Dynamically adjust font size based on the number of columns
        font_size = max(6, min(14, 22 - (sharpe_ratio_columns // 2)))
        
        heatmap_file_path = os.path.join(output_folder, f"{export_file_name}_SR_Heatmap.png")
        
        # Plot
        sharpe_ratios = self.sharpe_ratios.astype(float)
        mdds = self.mdds.astype(float)
        cumu_pnls = self.cumu_pnls.astype(float)
        calmar_ratios = self.calmar_ratios.astype(float)
        sortino_ratios = self.sortino_ratios.astype(float)
        
        plt.figure(figsize=(36, 16))
        
        plt.subplot(2, 3, 1)
        sns.heatmap(sharpe_ratios, annot=True, annot_kws={"size": font_size}, fmt=".2f", cmap="YlGnBu")
        plt.title("Sharpe Ratio Heatmap")
        plt.xlabel("Diff Threshold")
        plt.ylabel("Rolling Window")
        

        plt.subplot(2, 3, 2)
        sns.heatmap(mdds, annot=True, annot_kws={"size": font_size}, fmt=".2f", cmap="YlOrRd_r")
        plt.title("Maximum Drawdown Heatmap")
        plt.xlabel("Diff Threshold")
        plt.ylabel("Rolling Window")
        
        plt.subplot(2, 3, 3)
        sns.heatmap(cumu_pnls, annot=True, annot_kws={"size": font_size}, fmt=".2f", cmap="Blues")
        plt.title("Cumulative PnL Heatmap")
        plt.xlabel("Diff Threshold")
        plt.ylabel("Rolling Window")

        plt.subplot(2, 3, 4)
        sns.heatmap(calmar_ratios, annot=True, annot_kws={"size": font_size}, fmt=".2f", cmap="RdYlGn")
        plt.title("Calmar Ratio Heatmap")
        plt.xlabel("Diff Threshold")
        plt.ylabel("Rolling Window")
        

        plt.subplot(2, 3, 5)
        sns.heatmap(sortino_ratios, annot=True, annot_kws={"size": font_size}, fmt=".2f", cmap="YlGnBu")
        plt.title("Sortino Ratio Heatmap")
        plt.xlabel("Diff Threshold")
        plt.ylabel("Rolling Window")
        
        plt.tight_layout()
        plt.savefig(heatmap_file_path)
        plt.close()
        gc.collect()    # Explicit garbage collection

        print(f"[{self.__class__.__name__}] Saving sharpe ratios heatmap to '{heatmap_file_path}'")

        return heatmap_file_path
    
    
        
    def _print_best_params(self, coin: str, time_frame: str, model_name: str):
 
        best_sr_idx = self.sharpe_ratios.stack().idxmax() 
        best_sr = self.sharpe_ratios.stack().max()
        
        best_mdd_idx = self.mdds.stack().idxmin() 
        best_mdd = self.mdds.stack().min()
        
        best_calmar_idx = self.calmar_ratios.stack().idxmax()
        best_calmar = self.calmar_ratios.stack().max()
        
        best_sortino_idx = self.sortino_ratios.stack().idxmax() 
        best_sortino = self.sortino_ratios.stack().max()
        
        print("\n" + "="*50)
        print(f"Best Parameters for {coin} {time_frame} using {model_name}:")
        print("-"*50)
        print(f"Best Sharpe Ratio: {best_sr:.4f}")
        print(f"Best SR Parameters: rolling_window={best_sr_idx[0]}, diff_threshold={best_sr_idx[1]}")
        print(f"Best MDD: {best_mdd:.4f}")
        print(f"Best MDD Parameters: rolling_window={best_mdd_idx[0]}, diff_threshold={best_mdd_idx[1]}")
        print(f"Best Calmar Ratio: {best_calmar:.4f}")
        print(f"Best Calmar Parameters: rolling_window={best_calmar_idx[0]}, diff_threshold={best_calmar_idx[1]}")
        print(f"Best Sortino Ratio: {best_sortino:.4f}")
        print(f"Best Sortino Parameters: rolling_window={best_sortino_idx[0]}, diff_threshold={best_sortino_idx[1]}")
        print("="*50 + "\n")
