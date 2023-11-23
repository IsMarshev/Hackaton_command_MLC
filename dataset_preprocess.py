import pickle
from tqdm import tqdm
import pandas as pd
import os

path = os.getcwd()
filename = "train_data.pkl"


df = pd.read_parquet(path+'./dataset/DCS Бургер/hackaton2023_train.gzip')
name_list={}
for name, data in tqdm(df.groupby("customer_id")):
    date_list = {}
    history_len = 0
    for date, gruoped_data in data.drop("customer_id", axis=1).groupby("startdatetime"):
        date_list[date] = gruoped_data
    name_list[name] = date_list

with open(filename, 'wb') as f:
    pickle.dump(name_list, f)