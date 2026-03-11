import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

from utils.utils import STORAGE_DIR
import subprocess, os, glob
import pandas as pd

fp_cfm = ''
fp_input_dir = os.path.join(STORAGE_DIR, 'att_pcap')
fp_output_dir = os.path.join(project_root_dir, 'output')
subprocess.run([fp_cfm, fp_input_dir, fp_output_dir])
# subprocess.run(f'echo password | sudo -S {fp_cfm} {fp_input_dir} {fp_output_dir}', shell=True)

lst_fp_allcsv = glob.glob(os.path.join(project_root_dir, 'output', '*'))
lst_df = []
for fp_onecsv in lst_fp_allcsv:
    df_onecsv = pd.read_csv(fp_onecsv, header=0, index_col=None)
    lst_df.append(df_onecsv)
    subprocess.run(['rm', '-rf', fp_onecsv])

df = pd.concat(lst_df, axis=0)
df['Label'] == 'Attack'
print(df)

df.to_csv(os.path.join(fp_output_dir, 'raw_att.csv'), header=True, index=False)
