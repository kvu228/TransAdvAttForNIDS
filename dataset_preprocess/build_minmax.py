import pandas as pd

fp = ''
fp_fea = ''
path_to_save = ''

df = pd.read_csv(fp, header=0, index_col=None)

lst_fea = pd.read_csv(fp_fea, header=0, index_col=None).columns.tolist()

min_val = df[lst_fea].min(axis=0).values
max_val = df[lst_fea].max(axis=0).values

df_res = pd.DataFrame([min_val, max_val], columns=lst_fea)
df_res.to_csv(path_to_save, header=True, index=False)

