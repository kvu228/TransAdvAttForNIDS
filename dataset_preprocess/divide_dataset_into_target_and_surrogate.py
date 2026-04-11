import pandas as pd

fp = ''
df = pd.read_csv(fp, header=0, index_col=None)
dataset_for_target = ''
dataset_for_surrogate = ''

df_ben = df.loc[df['Label']=='Benign']
df_att = df.loc[df['Label']!='Benign']

df_ben = df_ben.sample(frac=1, replace=False, axis=0).reset_index(drop=True)
df_att = df_att.sample(frac=1, replace=False, axis=0).reset_index(drop=True)

num_ben = int(len(df_ben) * 0.5)
num_att = int(len(df_att) * 0.5)

df_t_ben = df_ben.iloc[:num_ben]
df_s_ben = df_ben.iloc[num_ben:]

df_t_att = df_att.iloc[:num_att]
df_s_att = df_att.iloc[num_att:]

df_t = pd.concat([df_t_ben, df_t_att], axis=0)
df_s = pd.concat([df_s_ben, df_s_att], axis=0)

df_t.to_csv(dataset_for_target, header=True, index=False)
df_s.to_csv(dataset_for_surrogate, header=True, index=False)