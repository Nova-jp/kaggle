import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

def smape(preds, target):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) を計算する。
    """
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    den = (np.abs(preds) + np.abs(target)) / 2
    res = num / den
    return 100 * np.sum(res) / n

def lgbm_smape(preds, train_data):
    """
    LightGBMのカスタム評価関数用のSMAPE。
    """
    labels = train_data.get_label()
    return 'smape', smape(preds, labels), False

def train_lgbm(x_train, y_train, x_val, y_val, params=None):
    """
    LightGBMモデルを訓練する。
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'seed': 42
        }
    
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
    
    model = lgb.train(
        params, 
        lgb_train, 
        valid_sets=[lgb_train, lgb_val], 
        feval=lgbm_smape,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
    )
    
    return model
