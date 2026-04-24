import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
from utils.utils import CustomDataset, init_net, STORAGE_DIR
from utils.MIFGSM_noSPTS import MIFGSM_noSPTS

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader

import os
import numpy as np
import pandas as pd

def getStepLen(fp_minmax):
    df_minmax = pd.read_csv(fp_minmax, header=0, index_col=None)
    step_len = 140 / (df_minmax.loc[1] - df_minmax.loc[0])
    step_len = step_len.replace([np.inf, -np.inf], 0)   # If column min value is equal to max value, dont change this column. Step length is set to 0.
    step_len.loc[step_len >= 1.] = 0.   # if column max value is samller than 140, dont change this column

    tensor_temp = torch.from_numpy(step_len.values).float()
    return tensor_temp

def main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_output):
    dev = torch.device("cuda")
    lr = 0.001
    epoch = 10
    batch_size = 128
    k = 0.9

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    step_len = getStepLen(fp_minmax).to(dev)

    net = init_net(model_type, f'{model_name}_{model_type}')
    net.to(dev)

    criterion = CrossEntropyLoss()
    optimizar = Adam(net.parameters(), lr=lr, betas=(0.99, 0.99))

    net.train()
    for i in range(epoch):
        cur_iter = 0
        for data, labels in dataloader:
            cur_iter += len(data)
            data, labels = data.to(dev), labels.to(dev)

            adv_data = MIFGSM_noSPTS(net, data.detach().clone(), labels, criterion, 7, step_len, dev)

            optimizar.zero_grad()
            loss = k * criterion(net(data), labels) + (1 - k) * criterion(net(adv_data), labels)
            loss.backward()
            optimizar.step()
            print(f'\r Normal adv training, Epoch:{i+1}/{epoch} Progress:{cur_iter}/{len(dataset)} Loss:{loss:.4f}', end='    ')
        print()
    torch.save(net.state_dict(), os.path.join(fp_output, f"normal_advtrain_{dataset_name}_{model_name}.pth"))

if __name__ == '__main__':

    dataset_name = 'ton' # or (ids18)
    model_name = 'mlp' # or (cnn, rescnn, lstm, Selfattention)
    model_type = 't'

    fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{model_type}.csv')
    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_minmax_{model_type}.csv')
    fp_dataset = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_sam_train_{model_type}.csv')
    fp_output = os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models')

    main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_output)
