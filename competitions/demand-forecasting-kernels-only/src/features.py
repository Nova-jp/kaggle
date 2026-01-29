import pandas as pd
import numpy as np

def create_date_features(df):
    """
    日付から基本的な特徴量と、円環構造をエンコードした特徴量を生成する。
    """
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    
    # 曜日の円環エンコード (Sin/Cos)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 月の円環エンコード (Sin/Cos)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def create_lag_features(df, lags=[91, 98, 105, 112, 119, 126, 182, 364]):
    """
    過去の売上実績に基づくラグ特徴量を生成する。
    """
    df = df.sort_values(by=['store', 'item', 'date'])
    for lag in lags:
        df['sales_lag_' + str(lag)] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.shift(lag))
    return df

def create_rolling_features(df, windows=[30, 90], shift=91):
    """
    過去の売上実績に基づく移動統計量（平均）を生成する。
    """
    df = df.sort_values(by=['store', 'item', 'date'])
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(['store', 'item'])['sales'].transform(
            lambda x: x.shift(shift).rolling(window=window).mean()
        )
    return df

def prepare_data(train_path, test_path=None):
    """
    データを読み込み、基本・ラグ・移動統計量の一連の特徴量を付与して返す。
    """
    train = pd.read_csv(train_path, parse_dates=['date'])
    if test_path:
        test = pd.read_csv(test_path, parse_dates=['date'])
        df = pd.concat([train, test], sort=False)
    else:
        df = train
    
    df = create_date_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    return df
