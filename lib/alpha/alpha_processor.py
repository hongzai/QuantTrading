from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lib.alpha.alpha_analysis import AlphaAnalysis
from lib.alpha.alpha_config import AlphaConfig
from lib.alpha.alpha_combine_method_enum import AlphaCombineMethodEnum

"""
To apply additional operation on alpha data (if necessary)
- AlphaAnalysis is taking the first rolling_windows as reference

Options:
1. To take original alpha column name
2. To combine 2 alpha columns with arithmetic operation
3. To apply custom formula
"""
class AlphaProcessor:
    def __init__(self, 
                 rolling_windows: List[int],
                 alpha_config: AlphaConfig=None,
                 enable_alpha_analysis: bool=False, 
                 enable_confirmation: bool=False,
                 output_folder: str=None):
        self.alpha_config = alpha_config
        self.output_folder = output_folder
        self.alpha_analysis = AlphaAnalysis(rolling_window=rolling_windows[0], enable_confirmation=enable_confirmation, output_folder=self.output_folder) if enable_alpha_analysis else None
    
    '''
    To process alpha based on AlphaConfig
    '''
    def run(self, data: pd.DataFrame) -> str:
        # Apply alpha changes
        if self.alpha_config.custom_formula:
            data[self.alpha_config.new_alpha_column_name] = self.alpha_config.custom_formula(data)      # Processing new alpha column
            new_alpha_column_name = self.alpha_config.new_alpha_column_name                             # Assign new alpha column name
        elif self.alpha_config.combine_method:
            data[self.alpha_config.new_alpha_column_name] = self.combine_alphas(data)                   # Processing new alpha column
            new_alpha_column_name = self.alpha_config.new_alpha_column_name                             # Assign new alpha column name
        else:
            new_alpha_column_name = self.alpha_config.new_alpha_column_name                             # Assign new alpha column name
            
        # Plot alpha analysis (if enabled)
        if self.alpha_analysis is not None:
            self.alpha_analysis.visualize_data(alpha_config=self.alpha_config, 
                                               data=data,
                                               new_alpha=data[new_alpha_column_name], 
                                               alpha1=data[self.alpha_config.alpha_column_1] if self.alpha_config.combine_method else None, 
                                               alpha2=data[self.alpha_config.alpha_column_2] if self.alpha_config.combine_method else None)
            self.alpha_analysis.visualize_data_model(data=data, 
                                                     column_name=new_alpha_column_name)
        
        return new_alpha_column_name

    '''
    To combine alpha and return new combined column name
    '''
    def combine_alphas(self, data: pd.DataFrame) -> pd.Series:
        alpha1 = data[self.alpha_config.alpha_column_1]
        alpha2 = data[self.alpha_config.alpha_column_2]
        
        combined_alpha = None
        if self.alpha_config.combine_method == AlphaCombineMethodEnum.ADD:
            combined_alpha = alpha1 + alpha2
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.SUBTRACT:
            combined_alpha = alpha1 - alpha2
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.MULTIPLY:
            combined_alpha = alpha1 * alpha2
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.DIVIDE:
            # 除法保护
            denominator = alpha2.replace(0, np.nan)  # 将0替换为NaN
            combined_alpha = alpha1 / denominator
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.DIVIDE_INVERSE:
            # 反向除法保护
            denominator = alpha1.replace(0, np.nan)
            combined_alpha = alpha2 / denominator
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.WEIGHTED_ADD:
            # 假设weights在alpha_config中定义
            w1, w2 = self.weights if hasattr(self, 'weights') else (1, 1)
            combined_alpha = (w1 * alpha1 + w2 * alpha2) / (w1 + w2)
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.LOG_RATIO:
            # 对数比率 (处理负值)
            eps = 1e-10  # 小数保护
            combined_alpha = np.log(np.abs(alpha1) + eps) - np.log(np.abs(alpha2) + eps)
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.PERCENT_DIFF:
            # 百分比差异
            combined_alpha = (alpha1 - alpha2) / np.abs(alpha2.replace(0, np.nan))
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.GEOMETRIC_MEAN:
            # 几何平均
            combined_alpha = np.sqrt(np.abs(alpha1 * alpha2)) * np.sign(alpha1 * alpha2)
        elif self.alpha_config.combine_method == AlphaCombineMethodEnum.HARMONIC_MEAN:
            # 调和平均
            denominator = (1/np.abs(alpha1) + 1/np.abs(alpha2)) / 2
            combined_alpha = np.sign(alpha1 * alpha2) / denominator
        else:
            self.alpha_config.methods = ", ".join([member.value for member in AlphaCombineMethodEnum])
            raise ValueError(f"Unknown self.combination_config.method: {self.alpha_config.combine_method.name}. Please choose from: {self.alpha_config.methods}")

        return combined_alpha

