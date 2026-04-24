import pandas as pd
import random

fp = '/home/lab401/mzs/programming/data/program_data/aa_in_nids_review/remove_fea_for_sm/storage/80_fea.csv'

lst_fea_all = pd.read_csv(fp, header=0, index_col=None).columns.to_list()

lst_fea_must_remove = ["Timestamp", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
                       "Flow IAT Min", "Active Mean", "Active Std", "Active Max",
                       "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"]

lst_fea_must_retain = ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Bwd Pkts', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
                'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd IAT Max', 'Fwd IAT Min', 'Flow Duration',
                'TotLen Fwd Pkts', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Fwd IAT Tot', 'Fwd IAT Mean',
                'Fwd IAT Std', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
                'Pkt Len Var', 'Pkt Size Avg', 'Flow Byts/s', 'Flow Pkts/s', 'Fwd Pkts/s',
                'Bwd Pkts/s', 'Subflow Fwd Byts']

lst_fea_will_be_removed = [e for e in lst_fea_all if e not in lst_fea_must_remove and e not in lst_fea_must_retain]

for i in range(0, 40, 3):
    lst_fea_random_removed = random.sample(lst_fea_will_be_removed, i)

    lst_fea_res = [e for e in lst_fea_all if e not in lst_fea_must_remove and e not in lst_fea_random_removed]

    # if len(lst_fea_res) == 63:

    df = pd.DataFrame([[0] * len(lst_fea_res)], columns=lst_fea_res)
    
    df.to_csv(f'/home/lab401/mzs/programming/data/program_data/aa_in_nids_review/remove_fea_for_sm/2_fea_csv/{len(lst_fea_res)}_fea.csv', header=True, index=False)




















