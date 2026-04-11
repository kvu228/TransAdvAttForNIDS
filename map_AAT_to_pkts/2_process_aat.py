import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd, numpy as np


fp_raw_aat = os.path.join(project_root_dir, 'output', 'raw_aat.csv')

df = pd.read_csv(fp_raw_aat, header=0, index_col=None)

# add new column called 'flow_id'
mask = df['Dst IP'] < df['Src IP']
flow_id_a = df['Src IP'] + '-' + df['Dst IP'] + '-' + df['Src Port'].astype(int).astype(str) + '-' + df['Dst Port'].astype(int).astype(str) + '-' + df['Protocol'].astype(int).astype(str)
flow_id_b = df['Dst IP'] + '-' + df['Src IP'] + '-' + df['Dst Port'].astype(int).astype(str) + '-' + df['Src Port'].astype(int).astype(str) + '-' + df['Protocol'].astype(int).astype(str)
df['flow_id'] = np.where(mask, flow_id_b, flow_id_a)

# remove duplicated flow_id 
df_2 = df[~df['flow_id'].duplicated(keep=False)]

print(df_2)
df_2.to_csv(os.path.join(project_root_dir, 'output', 'diff.csv'), header=True, index=False)