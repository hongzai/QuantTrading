from typing import Callable, List, Optional
import pandas as pd
from lib.alpha.alpha_combine_method_enum import AlphaCombineMethodEnum

class AlphaConfig:
    def __init__(self,
                alpha_column_1: str = None,
                alpha_column_2: str = None,
                combine_method: AlphaCombineMethodEnum = None,
                weights: List[float] = None,
                custom_formula: Callable[[pd.DataFrame], pd.Series] = None,
                new_alpha_column_name: str = None
        ):
        self.alpha_column_1 = alpha_column_1
        self.alpha_column_2 = alpha_column_2
        self.combine_method = combine_method
        self.weights = weights
        self.custom_formula = custom_formula
        self.new_alpha_column_name = new_alpha_column_name

    """
    Alternative constructor to use a single alpha column directly.

    :param alpha_column_name: Target column name.
    :return: Instance of AlphaConfig.
    """
    @classmethod
    def for_1_alpha(
        cls, 
        alpha_column_name: str
    ) -> 'AlphaConfig':
        return cls(alpha_column_1=alpha_column_name,
                   new_alpha_column_name=alpha_column_name)

    """
    Alternative constructor to combine two alpha columns using a specified method.

    :param alpha_column_1: First target column name.
    :param alpha_column_2: Second target column name.
    :param method: Combination method from AlphaCombineMethodEnum (e.g., ADD, SUBTRACT).
    :param weights: Weights for combination (only required for WEIGHTED_ADD method).
    :return: Instance of AlphaConfig.
    """
    @classmethod
    def for_combine_2_alphas(
        cls,
        alpha_column_1: str,
        alpha_column_2: str,
        combine_method: AlphaCombineMethodEnum,
        weights: Optional[List[float]] = None
    ) -> 'AlphaConfig':
        return cls(
            alpha_column_1=alpha_column_1,
            alpha_column_2=alpha_column_2,
            combine_method=combine_method,
            weights=weights,
            new_alpha_column_name=f"combined-{alpha_column_1}-{combine_method.name}-{alpha_column_2}"
        )

    """
    Alternative constructor to use a custom formula for combination.

    :param custom_formula: A callable that takes a pd.DataFrame and returns new column name.
    :param alpha_column_name: Target column name.
    :return: Instance of AlphaConfig.
    """
    @classmethod
    def for_custom_formula(
        cls,
        custom_formula: Callable[[pd.DataFrame], pd.Series],
        new_alpha_column_name: str
    ) -> 'AlphaConfig':
        return cls(custom_formula=custom_formula,
                   new_alpha_column_name=new_alpha_column_name)
