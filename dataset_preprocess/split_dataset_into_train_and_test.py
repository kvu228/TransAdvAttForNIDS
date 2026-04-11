import pandas as pd

fp_dataset = ''
path_to_save_training_dataset = ''
path_to_save_test_dataset = ''
split_ratio = 0.8

df = pd.read_csv(fp_dataset)

df_ben = df.loc[df['Label']=='Benign']
df_att = df.loc[df['Label']!='Benign']

df_ben = df_ben.sample(frac=1, replace=False, axis=0).reset_index(drop=True)
df_att = df_att.sample(frac=1, replace=False, axis=0).reset_index(drop=True)

num_ben = int(len(df_ben) * split_ratio)
num_att = int(len(df_att) * split_ratio)

df_training_ben = df_ben.iloc[:num_ben]
df_test_ben = df_ben.iloc[num_ben:]

df_training_att = df_att.iloc[:num_att]
df_test_att = df_att.iloc[num_att:]

df_training = pd.concat([df_training_ben, df_training_att], axis=0)
df_test = pd.concat([df_test_ben, df_test_att])

df_training.to_csv(path_to_save_training_dataset, header=True, index=False)
df_test.to_csv(path_to_save_test_dataset, header=True, index=False)