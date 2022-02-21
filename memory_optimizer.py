import pandas as pd
from typing import List


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=['float64']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    ints = df.select_dtypes(include=['int64']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')
    return df


def optimize_objects(df: pd.DataFrame, datetime_features: List[str]) -> pd.DataFrame:
    for col in df.select_dtypes(include=['object']):
        if col not in datetime_features:
            if not (type(df[col][0])==list):
                num_unique_values = len(df[col].unique())
                num_total_values = len(df[col])
                if float(num_unique_values) / num_total_values < 0.5:
                    df[col] = df[col].astype('category')
        else:
            df[col] = pd.to_datetime(df[col])
    return df


def memory_usage(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum()/1024**2


def optimize(df: pd.DataFrame, datetime_features: List[str] = []) -> pd.DataFrame:
    print('Memory usage before optimization: {:.3f} MB'.format(memory_usage(df)))
    df_optimized = optimize_floats(optimize_ints(optimize_objects(df, datetime_features)))
    print('Memory usage after optimization: {:.3f} MB'.format(memory_usage(df)))
    
    return df_optimized