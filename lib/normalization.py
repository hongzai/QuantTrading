import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lib.normalization_enum import NorModel

class Normalization:
    def __init__(self, model: NorModel, rolling_window: int = None, output_folder: str = "output"):
        self.model = model
        self.rolling_window = rolling_window
        self.output_folder = output_folder

    def normalize(self, data: pd.DataFrame, column_name: str) -> pd.Series:
        if self.model == NorModel.ZSCORE:
            normalized_data = self.calculate_zscore(data, column_name)
        elif self.model == NorModel.MINMAX:
            normalized_data = self.calculate_minmax(data, column_name)
        elif self.model == NorModel.ROBUST:
            normalized_data = self.calculate_robust(data, column_name)
        else:
            raise ValueError(f"Unknown normalization model: {self.model}")

        self.plot_distribution(normalized_data, column_name)
        return normalized_data

    def calculate_zscore(self, data: pd.DataFrame, column_name: str) -> pd.Series:
        mean = data[column_name].rolling(window=self.rolling_window).mean()
        std = data[column_name].rolling(window=self.rolling_window).std()
        return (data[column_name] - mean) / std

    def calculate_minmax(self, data: pd.DataFrame, column_name: str) -> pd.Series:
        min_val = data[column_name].rolling(window=self.rolling_window).min()
        max_val = data[column_name].rolling(window=self.rolling_window).max()
        return (data[column_name] - min_val) / (max_val - min_val)

    def calculate_robust(self, data: pd.DataFrame, column_name: str) -> pd.Series:
        median = data[column_name].rolling(window=self.rolling_window).median()
        q75 = data[column_name].rolling(window=self.rolling_window).quantile(0.75)
        q25 = data[column_name].rolling(window=self.rolling_window).quantile(0.25)
        iqr = q75 - q25
        return (data[column_name] - median) / iqr

    def plot_distribution(self, data: pd.Series, column_name: str):
        plt.figure(figsize=(10, 6))
        plt.hist(data.dropna(), bins=50, alpha=0.7, color='blue')
        plt.title(f'Distribution of {column_name} after {self.model.value} normalization')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Save the plot as a PNG file
        output_path = os.path.join(self.output_folder, f'Normalization_{self.model.value}_distribution.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        print(f"\nDistribution Chart saved at: {output_path}")
        user_input = input("\nContinue with this normalization? (y/n): ").lower()
        if user_input != 'y':
            print("User chose to terminate the process.")
            import sys
            sys.exit(0)
