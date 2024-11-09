import math
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import webbrowser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from lib.alpha.alpha_config import AlphaConfig
from lib.model.threshold_model_enum import ThresholdModelEnum
from lib.model.threshold_model_processor import ThresholdModelProcessor

"""
Plot useful insight (distribution/graph) for targeted alpha 
"""
class AlphaAnalysis:
    def __init__(self,
                 models: List[ThresholdModelEnum]=[ThresholdModelEnum.MA, ThresholdModelEnum.EMA, ThresholdModelEnum.ZSCORE, ThresholdModelEnum.MA_DIFF, ThresholdModelEnum.EMA_DIFF, ThresholdModelEnum.MINMAX, ThresholdModelEnum.ROBUST, ThresholdModelEnum.RSI], 
                 rolling_window: int = None, 
                 enable_confirmation: bool = False,
                 output_folder: str = "output"):
        self.models = models
        self.rolling_window = rolling_window
        self.enable_confirmation = enable_confirmation
        self.output_folder = output_folder

    '''
    To visualize the data using Plotly
    '''
    def visualize_data(self, 
                   alpha_config: AlphaConfig, 
                   data: pd.DataFrame, 
                   new_alpha: pd.Series, 
                   alpha1: pd.Series = None, 
                   alpha2: pd.Series = None) -> str:
        # Determine the number of subplots based on available alphas, including the histogram
        show_alpha_1 = alpha1 is not None
        show_alpha_2 = alpha2 is not None
        num_plots = 1 + int(show_alpha_1) + int(show_alpha_2) + 1  # +1 for new_alpha plot, +1 for missing row histogram

        # Create subplot titles
        subplot_titles = [f"Missing Records"]
        if show_alpha_1:
            subplot_titles.append(f'{alpha_config.alpha_column_1}')
        if show_alpha_2:
            subplot_titles.append(f'{alpha_config.alpha_column_2}')
        if alpha_config.custom_formula:
            subplot_titles.append(f'Custom Alpha ({alpha_config.new_alpha_column_name})')
        elif alpha_config.combine_method:
            subplot_titles.append(f'Combined Alpha ({alpha_config.alpha_column_1}-{alpha_config.combine_method.name}-{alpha_config.alpha_column_2})')
        else:
            subplot_titles.append(f'Alpha ({alpha_config.new_alpha_column_name})')

        # Initialize Plotly subplots
        fig = make_subplots(
            rows=num_plots,
            cols=1,
            shared_xaxes=True,
            subplot_titles=subplot_titles,
            vertical_spacing=0.05,
            row_heights=[0.2] + [0.8 / (num_plots - 1)] * (num_plots - 1) 
        )

        # --- Plot missing row histogram ---
        data = data.sort_values("start_time")
        expected_time_diffs = pd.Series(data['start_time']).diff().mode()[0]  # Most common interval as expected
        expected_times = pd.date_range(data['start_time'].iloc[0], data['start_time'].iloc[-1], freq=expected_time_diffs)
        missing_times = set(expected_times) - set(data['start_time'])
        missing_count = len(missing_times)

        fig.add_trace(
            go.Histogram(
                x=list(missing_times),
                nbinsx=50,
                name="Missing Timestamps",
                marker=dict(color='red'),
                opacity=0.7
            ),
            row=1,
            col=1
        )

        fig.update_yaxes(title_text="Missing Count", row=1, col=1)
        current_row = 2

        # --- Plot alpha1 if available ---
        if show_alpha_1:
            fig.add_trace(
                go.Scatter(
                    x=data['start_time'],
                    y=alpha1,
                    mode='lines',
                    name=f"Alpha ({alpha_config.alpha_column_1})",
                    line=dict(color='blue')
                ),
                row=current_row,
                col=1
            )
            fig.update_yaxes(title_text="Value", row=current_row, col=1)
            current_row += 1

        # --- Plot alpha2 if available ---
        if show_alpha_2:
            fig.add_trace(
                go.Scatter(
                    x=data['start_time'],
                    y=alpha2,
                    mode='lines',
                    name=f"Alpha ({alpha_config.alpha_column_2})",
                    line=dict(color='orange')
                ),
                row=current_row,
                col=1
            )
            fig.update_yaxes(title_text="Value", row=current_row, col=1)
            current_row += 1

        # Determine labels and output file name based on alpha configuration
        if alpha_config.custom_formula:
            label = f'Custom Alpha ({alpha_config.new_alpha_column_name})'
            output_file_name = f'CustomAlpha_{alpha_config.new_alpha_column_name}'
        elif alpha_config.combine_method:
            label = f'Combined Alpha ({alpha_config.alpha_column_1}-{alpha_config.combine_method.name}-{alpha_config.alpha_column_2})'
            output_file_name = f'CombinedAlpha_{alpha_config.alpha_column_1}_{alpha_config.combine_method.name}_{alpha_config.alpha_column_2}'
        else:
            label = f'Alpha ({alpha_config.new_alpha_column_name})'
            output_file_name = f'Alpha_{alpha_config.new_alpha_column_name}'

        # --- Plot new_alpha ---
        fig.add_trace(
            go.Scatter(
                x=data['start_time'],
                y=new_alpha,
                mode='lines',
                name=label,
                line=dict(color='green')
            ),
            row=current_row,
            col=1
        )
        fig.update_yaxes(title_text="Value", row=current_row, col=1)

        # Add a super title
        fig.update_layout(
            title=dict(
                text=label,
                x=0.5,
                font=dict(size=20)
            ),
            height=500 * num_plots,
            showlegend=False,
            template='plotly_white',
            margin=dict(t=100, b=50, l=50, r=50)
        )

        output_path_html = os.path.join(self.output_folder, f'{output_file_name}.html')
        Path(os.path.dirname(output_path_html)).mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path_html)

        # Optionally, save the plot as a PNG file using Kaleido
        output_path_png = os.path.join(self.output_folder, f'{output_file_name}.png')
        try:
            fig.write_image(output_path_png)
        except ValueError as e:
            print(f"Could not save PNG image. Ensure Kaleido is installed. Error: {e}")

        # Automatically open the HTML output file in the default web browser
        try:
            webbrowser.open('file://' + os.path.realpath(output_path_html))
        except Exception as e:
            print(f"Could not open the file automatically: {e}")

        # Prompt Confirmation
        if self.enable_confirmation:
            print(f"\nAlpha Chart saved at: {output_path_html}")
            user_input = input("\nContinue? (y/n): ").strip().lower()
            if user_input != 'y':
                print("User chose to terminate the program.")
                sys.exit(0)

        return output_path_html
    
    
    '''
    To visualize the data with model using Plotly
    '''
    def visualize_data_model(self, data: pd.DataFrame, column_name: str) -> pd.Series:
        normalized_data = pd.DataFrame(index=data.index)
        
        for model in self.models:
            new_data = data.copy()
            new_column_name = ThresholdModelProcessor(model).run(new_data, column_name, "", "", self.rolling_window, 0)
            
            # Special handling for model that return original column name (MA, EMA)
            if model==ThresholdModelEnum.MA:
                new_column_name = f'{column_name}-ma'
            elif model==ThresholdModelEnum.MA:
                new_column_name = f'{column_name}-ema'
                
            normalized_data[model.value] = new_data[new_column_name]
            
        # Ensure 'close' column exists
        if 'close' not in data.columns:
            raise ValueError("The input data must contain a 'close' column.")

        # Plot all models' distributions in one figure
        normalized_data["close"] = new_data["close"]
        self.plot_data_model(normalized_data, column_name)
        
        return normalized_data


    def plot_data_model(self, data: pd.DataFrame, column_name: str):
        num_models = len(self.models)
        num_cols = 2  # Two diagrams per row: Line Chart and Histogram
        num_rows = math.ceil(num_models)  # One row per model

        # Define the amount of horizontal and vertical spacing between subplots
        horizontal_spacing = 0.10  # Reduced horizontal spacing for less gap
        vertical_spacing = 0.05    # Reduced vertical spacing for less gap

        # Create subplot titles by interleaving Line Chart and Histogram titles
        subplot_titles = []
        for model in self.models:
            subplot_titles.append(f'{model.value.upper()} vs Close Price')
            subplot_titles.append(f'Distribution of {model.value.upper()} Normalization')

        # Create subplots: for each model, one line chart and one histogram
        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": True}, {"type": "histogram"}] for _ in range(num_rows)],
            horizontal_spacing=horizontal_spacing,  # Adjusted horizontal spacing
            vertical_spacing=vertical_spacing       # Adjusted vertical spacing
        )

        for i, model in enumerate(self.models):
            row = i + 1
            col_line = 1
            col_hist = 2
            model_value = model.value

            # --- Plot Line Chart with Twin Y-Axes ---
            # Plot Close Price on primary y-axis
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["close"],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                    #line=dict(color='rgba(173, 216, 230, 1.0)')  # Light blue
                ),
                row=row,
                col=col_line,
                secondary_y=False
            )

            # Plot Model Data on secondary y-axis
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[model_value],
                    mode='lines',
                    name=f'{model_value.upper()}',
                    #line=dict(color='red')
                    line=dict(color='lightcoral')  # Changed to lighter red
                ),
                row=row,
                col=col_line,
                secondary_y=True
            )

            # --- Plot Histogram ---
            fig.add_trace(
                go.Histogram(
                    x=data[model_value].dropna(),
                    nbinsx=50,
                    name=f'{model_value.upper()} Distribution',
                    marker=dict(color='green'),
                    opacity=0.7
                ),
                row=row,
                col=col_hist
            )

        # Update layout for better visuals and center the main title
        fig.update_layout(
            title=dict(
                text=f'Alpha Models ({column_name}) with RW {self.rolling_window}<br>[RW is based on first rolling_windows parameter]',
                x=0.5,  # Center the title horizontally
                xanchor='center',  # Anchor the title at the center
                y=0.99,  # Position the title at the very top
                yanchor='top',  # Anchor the title's y position to the top
                font=dict(size=20)
            ),
            height=400 * num_rows,  # Adjust height based on number of rows (reduced from 600)
            showlegend=False,  # We'll handle legends individually
            template='plotly_white',
            margin=dict(t=100, b=50, l=50, r=50),  # Adjust margins
            title_x=0.5  # Ensures the title is centered
        )

        # Update subplot titles font size for better fit
        fig.update_layout(
            font=dict(size=12)  # Reduced font size for subplot titles
        )

        # Update x-axis for all subplots
        for i in range(1, num_rows * num_cols + 1):
            fig.update_xaxes(title_text="Index", row=(i - 1) // num_cols + 1, col=(i - 1) % num_cols + 1)

        # Update y-axes titles and colors for each row
        for i, model in enumerate(self.models):
            row = i + 1
            # Primary y-axis for Close Price
            fig.update_yaxes(title_text="Close Price", row=row, col=1, secondary_y=False, color='blue')
            # Secondary y-axis for Model Data
            fig.update_yaxes(title_text=model.value.upper(), row=row, col=1, secondary_y=True, color='red')
            # y-axis for Histogram
            fig.update_yaxes(title_text="Frequency", row=row, col=2)

        # Save the interactive plot as an HTML file
        output_file = os.path.join(self.output_folder, 'Alpha_analysis.html')
        os.makedirs(self.output_folder, exist_ok=True)
        fig.write_html(output_file)

        # --- Automatically open the output file in the default browser ---
        try:
            webbrowser.open('file://' + os.path.realpath(output_file))
        except Exception as e:
            print(f"Could not open the file automatically: {e}")

        # --- Prompt Confirmation ---
        if self.enable_confirmation:
            print(f"\nAlpha Analysis for '{column_name}' saved at: {output_file}")
            user_input = input("\nContinue with this alpha? (y/n): ").strip().lower()
            if user_input != 'y':
                print("User chose to terminate the process.")
                sys.exit(0)