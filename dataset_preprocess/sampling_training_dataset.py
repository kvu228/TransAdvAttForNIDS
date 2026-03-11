import pandas as pd
import os, sys

# Project root and STORAGE_DIR for output paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.utils import STORAGE_DIR

dataset_dir = os.path.join(STORAGE_DIR, 'dataset')

for model_type in ['s', 't']:
    fp = os.path.join(dataset_dir, f'ids18_train_{model_type}.csv')
    path_to_save_dataset = os.path.join(dataset_dir, f'ids18_sam_train_{model_type}.csv')

    if not os.path.isfile(fp):
        print(f'Skip: input not found: {fp}')
        continue

    print(f'Loading {fp} ...')
    df = pd.read_csv(fp, header=0, index_col=None)

    df_ben = df.loc[df['Label']=='Benign']
    df_att = df.loc[df['Label']!='Benign']

    if len(df_ben) > len(df_att):
        num = len(df_ben)
        df_att = df_att.sample(num, replace=True, axis=0).reset_index(drop=True)
    elif len(df_ben) < len(df_att):
        num = len(df_att)
        df_ben = df_ben.sample(num, replace=True, axis=0).reset_index(drop=True)
    else:
        pass

    df_res = pd.concat([df_ben, df_att], axis=0)
    df_res.to_csv(path_to_save_dataset, header=True, index=False)
    print(f'Saved: {path_to_save_dataset}')
