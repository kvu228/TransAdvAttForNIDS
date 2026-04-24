import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
from utils.utils import CustomDataset, init_net, STORAGE_DIR
from utils.MIFGSM_forAdvTrain import MIFGSM_forAdvTrain
from utils.NIFGSM_forAdvTrain import NIFGSM_forAdvTrain
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
import pandas as pd
import os
import numpy as np

def get_mask(list_col:list, batch_size):
    df_temp = pd.DataFrame([[0.] * len(list_col)], columns=list_col)
    df_temp["Fwd Pkt Len Max"] = 1.
    df_temp["Fwd Pkt Len Min"] = 1.
    df_temp["Fwd IAT Max"] = 1.
    df_temp["Fwd IAT Min"] = 1.
    tensor_temp = torch.from_numpy(df_temp.loc[0].values).repeat(batch_size, 1)
    return tensor_temp

def main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_output, attack_name='MIFGSM'):

    dev = torch.device('cuda')
    k = 0.8
    lr = 0.001
    epoch = 10
    batch_size = 128

    df_raw_data = pd.read_csv(fp_dataset, header=0, index_col=None)
    df_raw_data['label'] = 1
    df_raw_data.loc[df_raw_data['Label'] == 'Benign', 'label'] = 0
    list_fea = pd.read_csv(fp_fea, header=0, index_col=None).columns.tolist()
    df_minmax = pd.read_csv(fp_minmax, header=0, index_col=None)

    net = init_net(model_type, f'{model_name}_{model_type}')
    net.to(dev)

 
    criterion = CrossEntropyLoss()
    optimizar = Adam(net.parameters(), lr=lr, betas=(0.99, 0.99))

    net.train()
    for i in range(epoch):
        # shuffle data
        df_raw_data = df_raw_data.sample(frac=1., replace=False).reset_index(drop=True)
        mask = get_mask(list_fea, batch_size)
        mask = mask.to(dev)

        cur_iter, pos_row = 0, 0
        while pos_row < len(df_raw_data):
            df_part_flow = df_raw_data.iloc[pos_row: pos_row + batch_size]
            pos_row += len(df_part_flow)

            # get label
            labels = torch.from_numpy(df_part_flow['label'].values).to(dev)

            # get normalized flow, para1
            df_part_flow_66fea = df_part_flow[list_fea]
            df_part_flow_66fea = ((df_part_flow_66fea - df_minmax.loc[0]) / (df_minmax.loc[1] - df_minmax.loc[0])).fillna(0)
            tensor1 = torch.from_numpy(df_part_flow_66fea.values).float().to(dev)

            # get adv flow
            if len(df_part_flow) < batch_size:
                mask = mask[0:len(df_part_flow)]
            adv_train_fn = NIFGSM_forAdvTrain if attack_name == 'NIFGSM' else MIFGSM_forAdvTrain
            df_adv_flow = adv_train_fn(net, criterion, df_part_flow[list_fea], labels, mask, 7, 140, dev, df_minmax.loc[0], df_minmax.loc[1])
            df_adv_flow = ((df_adv_flow - df_minmax.loc[0]) / (df_minmax.loc[1] - df_minmax.loc[0])).fillna(0).replace([np.inf, -np.inf], [1.0, -1.])
            pos = df_part_flow['label'] == 0
            df_adv_flow.loc[pos] = df_part_flow_66fea.loc[pos]
            tensor2 = torch.from_numpy(df_adv_flow.values).float().to(dev)
            tensor2 = torch.clamp(tensor2, 0., 1.)

            optimizar.zero_grad()
            loss = k * criterion(net(tensor1), labels) + k * criterion(net(tensor2), labels)
            loss.backward()
            optimizar.step()
            print(f'\r Adv training with SPTS, Epoch:{i+1}/{epoch},Progress:{pos_row},Loss:{loss:.4f}', end='    ')

        print()
    suffix = f"_{attack_name}" if attack_name != 'MIFGSM' else ""
    torch.save(net.state_dict(), os.path.join(fp_output, f"advtrain_withSPTS_{dataset_name}_{model_name}{suffix}.pth"))

if __name__ == '__main__':

    dataset_name = 'ton' # or (ids18)
    model_name = 'mlp' # or (cnn, rescnn, lstm, Selfattention)
    model_type = 't'

    fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{model_type}.csv')
    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_minmax_{model_type}.csv')
    fp_dataset = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_sam_train_{model_type}.csv')
    fp_output = os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models')

    attack_name = 'MIFGSM'  # or 'NIFGSM'
    main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_output, attack_name)