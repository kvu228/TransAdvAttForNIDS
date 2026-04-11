import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

from utils.utils import load_net, STORAGE_DIR
from utils.MIFGSM import MIFGSM
from utils.SIM import SIM
from utils.VMIFGSM import VMIFGSM
from utils.DGM import DGM
from utils.NIFGSM import NIFGSM

import pandas as pd
from torch.nn import CrossEntropyLoss
import torch
import os, glob
import subprocess

def get_mask(list_col:list, batch_size):
    df_temp = pd.DataFrame([[0.] * len(list_col)], columns=list_col)
    df_temp["Fwd Pkt Len Max"] = 1.
    df_temp["Fwd Pkt Len Min"] = 1.
    df_temp["Fwd IAT Max"] = 1.
    df_temp["Fwd IAT Min"] = 1.
    tensor_temp = torch.from_numpy(df_temp.loc[0].values).repeat(batch_size, 1)
    return tensor_temp

def main(dataset_name, model_name, model_type, attack_name, iteration, step_size, fp_model, fp_raw_att, fp_fea, fp_minmax, path_to_save_aat):

    print("Loading variable...")
    batch_size = 128
    dev = torch.device("cuda")
    att = None
    if attack_name == 'MIFGSM':
        att = MIFGSM
    if attack_name == 'SIM':
        att = SIM
    if attack_name == 'VMIFGSM':
        att = VMIFGSM
    if attack_name == 'DGM':
        att = DGM
    if attack_name == 'NIFGSM':
        att = NIFGSM

    lossfn = CrossEntropyLoss()

    if os.path.exists(path_to_save_aat):
        subprocess.run(['rm', '-r', path_to_save_aat])

    list_sm_col = pd.read_csv(fp_fea, header=0, index_col=None).columns.tolist()
    mask = get_mask(list_sm_col, batch_size)
    mask = mask.to(dev)

    df_minmax = pd.read_csv(fp_minmax, header=0, index_col=None)[list_sm_col]

    surrogate_model = load_net(f'{dataset_name}_{model_name}_{model_type}', fp_model)
    surrogate_model.to(dev)

    if 'lstm' in model_name:
        surrogate_model.train()
    else:
        surrogate_model.eval()

    fea_res = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
               'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd IAT Max', 'Fwd IAT Min'] # Do not need save all features this time

    fea_4 = ['Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd IAT Max', 'Fwd IAT Min'] # 4 level 1 features

    cunt, cunt2 = 0, 0
    for raw_flow in pd.read_csv(fp_raw_att, header=0, index_col=None, chunksize=batch_size):

        cunt += len(raw_flow)
        cunt2 += 1
        
        if len(raw_flow) < batch_size:
            mask = mask[0: len(raw_flow)]

        raw_flow2 = raw_flow.copy(deep=True)
        df_flow = raw_flow[list_sm_col]
        df_adv_flow, _ = att(surrogate_model, lossfn, df_flow, None, mask, 7, 140, dev, df_minmax.loc[0], df_minmax.loc[1])
        raw_flow[list_sm_col] = df_adv_flow

        dif = raw_flow[fea_4].values - raw_flow2[fea_4].values      # compute diff between adv traffic and raw traffic

        pos = ~(raw_flow2.reset_index(drop=True) == raw_flow.reset_index(drop=True)).all(axis=1)        # remove the traffic not be modified
        raw_flow[fea_4] = dif

        df_res = raw_flow.reset_index(drop=True).loc[pos]
        df_res = df_res[fea_res]            # only save the features used to modify pkts

        isexist = os.path.exists(path_to_save_aat)
        df_res.to_csv(path_to_save_aat, mode='a', header=not isexist, index=False)
        print(f'\r {dataset_name}, {model_name}, {attack_name}, Progress:{cunt}', end='')
    print()

if __name__ == "__main__":
    dataset_name = 'ton'
    model_name = 'mlp' # or (cnn, rescnn, lstm, Selfattention)
    model_type = 's'

    attack_name = 'MIFGSM' # or ('SIM', 'VMIFGSM', 'DGM')
    iteration = 7
    step_size = 140

    fp_model = os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models', f'{dataset_name}_{model_name}_{model_type}.pth')
    fp_raw_att = os.path.join(project_root_dir, 'output', 'raw_att.csv')
    fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{model_type}.csv')
    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_minmax_{model_type}.csv')
    path_to_save_aat = os.path.join(project_root_dir, 'output', 'raw_aat.csv')

    main(dataset_name, model_name, model_type, attack_name, iteration, step_size, fp_model, fp_raw_att, fp_fea, fp_minmax, path_to_save_aat)