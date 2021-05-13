import pandas as pd
import numpy as np
import re
import os
import config
import utils

def load_dataset(path, dates_column=None):
    """
    A function that loads the dataset.
    Args:
        path (String): Path to the csv dataset file.
        dates_column (List): List with one item, the date column name.
    Returns:
        Dataframe: A pandas dataframe
    """
    if dates_column:
        df = pd.read_csv(path, parse_dates=dates_column)
    else:
        df = pd.read_csv(path)

    return df

def next_output_file_name(path):
    """
    A function the returns the name of the next output file.
    Args:
        path (String): Path to the output file directory.
    Returns:
        String: The name of the output CSV file.
    """

    if len(os.walk(path).__next__()[2]) > 0:
        next_file = len(os.walk(path).__next__()[2]) + 1
    else:
        next_file = 1
    next_file_name = "submission_" + str(next_file) + ".csv"
    return next_file_name

def create_bins_and_day_column(dataframe, name):
    """
    A function that creates a time_bin and a day name column in a dataframe.
    Args:
        dataframe (Dataframe): A pandas dataframe
        name (String): The name of the pandas dataframe. 
        Used to distiguish between the test and train 
        dataframes because they have different date column names.
    Returns:
        Dataframe: A dataframe updated with a "time_bin" and "day" column.
    """

    if name == 'data':
        dataframe['time_bin'] = pd.cut(
            dataframe.datetime.dt.hour, 
            config.TIME_BINS, 
            labels=config.BIN_LABELS,
            right=False
        )
        dataframe['day'] = dataframe['datetime'].dt.day_name()
    else:
        dataframe['time_bin'] = pd.cut(
            dataframe.date.dt.hour, 
            config.TIME_BINS, 
            labels=config.BIN_LABELS,
            right=False
        )
        dataframe['day'] = dataframe['date'].dt.day_name()
    return dataframe



