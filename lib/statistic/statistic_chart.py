import gc
from matplotlib import pyplot as plt
import pandas as pd
import os

'''
This class is used to generate chart
'''
class StatisticChart():
    def __init__(self, rolling_windows: list, diff_thresholds: list):
        self.rolling_windows = rolling_windows
        self.diff_thresholds = diff_thresholds
        
    def export_chart(self, file_path: str, alpha_column_name: str, data: pd.DataFrame):
        # --- Process data ---
        columns_to_convert = ['drawdown', 'position', 'close', 'cumu_PnL', alpha_column_name, 'SR', 'MDD', 'AR', 'CR', 'Trading Fee', 'Trade Count']
        
        if 'lower_threshold' in data.columns and 'upper_threshold' in data.columns:
            columns_to_convert.append('lower_threshold')
            columns_to_convert.append('upper_threshold')
            
        for col in columns_to_convert:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # --- Try to load signals data ---
        signals_dir = os.path.join(os.path.dirname(file_path), 'signals')
        signals_file = os.path.join(signals_dir, os.path.basename(file_path).replace('.png', '_signals.pkl'))
        
        if os.path.exists(signals_file):
            try:
                signals_data = pd.read_pickle(signals_file)
                data['signals'] = data['position'].diff().fillna(0)
                data.loc[data['signals'] > 0, 'signals'] = 1    # long buy signal
                data.loc[data['signals'] < 0, 'signals'] = -1   # short sell signal
            except Exception as e:
                print(f"Warning: Could not load signals data: {e}")

        # --- Plot the close price and cumu PnL chart ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

        # Add y-axis for price
        ax1.plot(data.index, data['close'], label="Price", color='blue')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Close Price')
        ax1.legend(loc='upper left')

        # mark the buy and sell signals on the price line
        if 'signals' in data.columns:
            buy_signals = data[data['signals'] == 1]
            sell_signals = data[data['signals'] == -1]
            
            ax1.scatter(buy_signals.index, buy_signals['close'], 
                       marker='^', color='green', s=100, label='Buy Signal')
            ax1.scatter(sell_signals.index, sell_signals['close'], 
                       marker='v', color='red', s=100, label='Sell Signal')
            
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper left')

        # Add second y-axis for cumu_PnL
        ax1_2 = ax1.twinx()
        ax1_2.plot(data.index, data['cumu_PnL'], label='Cumu PnL', color='orange')
        ax1_2.set_ylabel('Cumu PnL')
        ax1_2.legend(loc='upper left', bbox_to_anchor=(0, 0.94))

        # the signal mark on the cumu_PnL chart
        # if 'signals' in data.columns:
        #     buy_signals = data[data['signals'] == 1]
        #     sell_signals = data[data['signals'] == -1]

        #     # Mark buy and sell signals on the cumu PnL chart
        #     ax1_2.scatter(buy_signals.index, buy_signals['cumu_PnL'], marker='^', color='blue', label='Buy Signal')
        #     ax1_2.scatter(sell_signals.index, sell_signals['cumu_PnL'], marker='v', color='red', label='Sell Signal')
        #     ax1_2.legend(loc='upper left', bbox_to_anchor=(0, 0.94))

        ax1.set_title(file_path.replace('.png', ''))
        ax1.grid(True)

        # Add additional information as text
        info_text = (
            f"SR: {data['SR'].iloc[0]:.2f}\n"
            f"MDD: {data['MDD'].iloc[0]:.2f}\n"
            f"AR: {data['AR'].iloc[0]:.2f}\n"
            f"CR: {data['CR'].iloc[0]:.2f}\n"
            f"TF: {data['Trading Fee'].iloc[0]}\n"
            f"TC: {data['Trade Count'].iloc[0]}"
        )
        ax1.text(0.01, 0.85, info_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='center', ha='left', bbox=dict(facecolor='white', alpha=0.5))

        ax1.set_xlabel('Index')
        ax1.set_ylabel('Close Price')
        ax1.legend(loc='upper left')

        # --- Plot the signal and threshold ---
        if 'lower_threshold' in data.columns and 'upper_threshold' in data.columns:
            ax2.plot(data.index, data[alpha_column_name], label=alpha_column_name, color='pink')
            ax2.plot(data.index, data['lower_threshold'], label='Lower Threshold', color='red')
            ax2.plot(data.index, data['upper_threshold'], label='Upper Threshold', color='green')
            ax2.set_xlabel('Index')
            ax2.set_ylabel('Value')
            ax2.set_title('Alpha and Thresholds')
            ax2.legend(loc='upper left')
            ax2.grid(True)

            # Focus on the range of lower and upper thresholds
            mid_threshold = (abs(data['lower_threshold'].min()) + abs(data['upper_threshold'].max())) / 2
            ax2.set_ylim([data['lower_threshold'].min() - (mid_threshold * 2.0), data['upper_threshold'].max() + (mid_threshold * 2.0)])

        # --- Plot the drawdown chat ---
        ax3.fill_between(data.index, data['drawdown'], color='red', alpha=0.5, label='Drawdown')
        ax3.set_xlabel('Index')
        ax3.set_ylabel('Drawdown')
        ax3.axhline(y=data['drawdown'].mean(), color='red', linestyle='--', label='Drawdown mean')

        # Draw lines
        #ax3.axhline(y=-0.0372, color='purple', linestyle='--', label='add = -3.72%')
        #ax3.axhline(y=-0.0768, color='blue', linestyle='--', label='ladd = -7.68%')
        #ax3.axhline(y=-0.1002, color='orange', linestyle='--', label='asdd = -10.02%')

        # Legends
        ax3.legend(loc='lower left')
        ax3.set_title('Drawdown Over Time')
        ax3.grid(True)

        # --- Display the plot ---
        plt.tight_layout()
        #plt.show()
        plt.savefig(file_path)
        fig.clear()
        plt.close(fig)  # Close the figure to release memory
        gc.collect()    # Explicit garbage collection