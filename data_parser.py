import numpy as np
import os
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def to_df(path, **kwargs):
    to_timestamp = kwargs.get("to_timestamp", False)
    
    if not((os.path.isfile(path)) and os.access(path, os.R_OK)):
        print("ERROR: File not found")
        return None
    
    df = pd.read_table(path, delimiter='\t', skip_blank_lines=True, na_values=" ")
    df.drop(columns=list(df.columns)[-1], inplace=True)
    df = df.loc[df["Date"] != " "]

    if to_timestamp:
        for i in range(len(df)-1):
            date_list = df["Date"][i].split("/")
            time_list = df["Time"][i].split(":")
            df["Time"][i] = pd.Timestamp(year=int(date_list[0]), month=int(date_list[1]), day=int(date_list[2]), hour=int(time_list[0]), minute=int(time_list[1]), second=int(time_list[2]))
    
    return df


def convert_datetime(df):
    for i in range(len(df)):
        date = df["Date"][i].split("/")
        time = df["Time"][i].split(":")
        df["Time"][i] = pd.Timestamp(year=int(date[0]), month=int(date[1]), day=int(date[2]), hour=int(time[0]), minute=int(time[1]), second=int(time[2]))
