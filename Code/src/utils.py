"""Utility functions for data processing and visualization."""
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def add_class_name_prefix(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Adds a class name prefix to the given filename column in a DataFrame.

    Args:
        df (pd.DataFrame): Input pandas DataFrame.
        col_name (str): Name of the column containing filenames.

    Returns:
        pd.DataFrame: DataFrame with modified column.
    """
    df_copy = df.copy()
    df_copy[col_name] = df_copy[col_name].apply(
        lambda x: x[re.search("_", x).start() + 1: re.search("_", x).start() + 2]
                  + "/" + x
    )
    return df_copy