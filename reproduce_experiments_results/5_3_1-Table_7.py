"""Mục 5.3.1 — thống kê nhãn (TON_IoT raw attack traffic).

Giống ``5_3_1-Table_6.py`` cho ``ton_raw_att.csv`` (đếm nhãn + tổng mẫu).
"""
import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import STORAGE_DIR

fp = os.path.join(STORAGE_DIR, 'dataset', 'ton_raw_att.csv')
df = pd.read_csv(fp, header=0, index_col=None)
res = df['Label'].value_counts().to_dict()
res['All'] = len(df)

for key in res.keys():
    print(f'{key}: {res[key]}')