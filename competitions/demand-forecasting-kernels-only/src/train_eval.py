import os
import sys
import pandas as pd
import numpy as np

# プロジェクトルートを作業ディレクトリに追加
sys.path.append("/work/competitions/demand-forecasting-kernels-only/")

from src.features import prepare_data
from src.models import train_lgbm, smape

def main():
    print("--- データの準備中 ---")
    DATA_DIR = "/work/competitions/demand-forecasting-kernels-only/data"
    train_path = f"{DATA_DIR}/train.csv"
    
    # データのロードと特徴量生成
    df = prepare_data(train_path)
    
    # 時系列での分割（最後の3ヶ月を検証用に）
    split_date = '2017-10-01'
    df_train = df[df['date'] < split_date].dropna()
    df_val = df[df['date'] >= split_date].dropna()
    
    # 使用する特徴量の定義
    # 数学修士の直感：円環構造(Sin/Cos)を採用
    features = [
        'store', 'item', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
        'sales_lag_91', 'sales_lag_98', 'sales_lag_105', 'sales_lag_112',
        'rolling_mean_30', 'rolling_mean_90'
    ]
    
    x_train = df_train[features]
    y_train = df_train['sales']
    x_val = df_val[features]
    y_val = df_val['sales']
    
    print(f"訓練データ数: {len(x_train)}")
    print(f"検証データ数: {len(x_val)}")
    print(f"特徴量数: {len(features)}")
    
    print("\n--- モデルの訓練開始 (Cyclic Encoding) ---")
    model = train_lgbm(x_train, y_train, x_val, y_val)
    
    # 評価
    preds = model.predict(x_val)
    final_score = smape(preds, y_val)
    
    print("\n" + "="*30)
    print(f"最終検証SMAPEスコア: {final_score:.6f}")
    print("="*30)
    
    # 重要度の確認
    import lightgbm as lgb
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values(by='importance', ascending=False)
    
    print("\n--- 特徴量重要度 (Top 5) ---")
    print(importance.head(5))

if __name__ == "__main__":
    main()
