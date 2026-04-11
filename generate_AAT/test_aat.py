import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import pandas as pd
import torch
from torch.utils.data import DataLoader

def main(dataset_name, model_name, model_type, fp_model, fp_att, fp_fea, fp_minmax):
    dev = torch.device('cuda')
    batch_size = 128
    
    dataset = CustomDataset(fp_att, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    net = load_net(f'{dataset_name}_{model_name}_{model_type}', fp_model)
    net.to(dev)

    net.eval()
    TP, FP, TN, FN, curr_iter = 0, 0, 0, 0, 0
    for flows, labels in dataloader:
        flows, labels = flows.to(dev), labels.to(dev)
        curr_iter += len(labels)

        with torch.no_grad():
            pred = net(flows).argmax(1)

        TP += ((pred == 1) & (labels == 1)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()
        acc = (TP / (TP + FN)) * 100
        print(f"\rTarget model:{model_name}, Acc: {acc:.1f}", end="")
    print()
    return acc

if __name__ == '__main__':
    dataset_name = 'ton' # or (ids18)
    model_name = 'mlp' # or (cnn, rescnn, lstm, Selfattention)
    model_type = 't'

    fp_model = os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models', f'{dataset_name}_{model_name}_{model_type}.pth')
    fp_att = os.path.join(STORAGE_DIR, 'custom', 'output', 'aat.csv')
    fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{model_type}.csv')
    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_minmax_{model_type}.csv')

    main(dataset_name, model_name, model_type, fp_model, fp_att, fp_fea, fp_minmax)


