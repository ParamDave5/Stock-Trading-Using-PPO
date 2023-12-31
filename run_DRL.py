# common library
import pandas as pd
import numpy as np
import time
 

# model
from Finrl.model.models import run_ppo

import os
def demo():
    print("import done")
def run_model() -> None:
    """Train the model."""

    # read and preprocess data
    preprocessed_path = "final2.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)

    print(data.head())
    print(data.size)
    print(data.columns)

    # 2015/10/01 is the date that validation starts
    # 2016/01/01 is the date that real trading starts
    # unique_trade_date needs to start from 2015/10/01 for validation purpose

    unique_trade_date = data[(data.date > 20151001)&(data.date <= 20200707)].date.unique()
    print(unique_trade_date)

    # rebalance_window is the number of months to retrain the model
    # validation_window is the number of months to validation the model and select for trading
    rebalance_window = 63
    validation_window = 63
    
    ## Ensemble Strategy

    run_ppo(df=data, 
            unique_trade_date= unique_trade_date,
            rebalance_window = rebalance_window,
            validation_window=validation_window)

###needs some more trianing code. 

    #_logger.info(f"saving model version: {_version}")

if __name__ == "__main__":
    run_model()
