import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
import pandas as pd
from utils.utils import CustomDataset, load_net, STORAGE_DIR
import torch
from torch.utils.data import DataLoader

def main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_model):
    dev = torch.device('cuda')
    batch_size = 64

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
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
        FP += ((pred == 1) & (labels == 0)).sum().item()
        TN += ((pred == 0) & (labels == 0)).sum().item()
        FN += ((pred == 0) & (labels == 1)).sum().item()

        acc = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) !=0 else 0
        pre = TP / (TP + FP) if (TP + FP) != 0 else 0
        rec = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (pre * rec) / (pre + rec) if pre + rec != 0 else 0

        print(f"\r{dataset_name}_{model_name}_{model_type}, Prog:{curr_iter}/{len(dataset)}, Acc: {acc:.3f} Pre: {pre:.3f} Rec: {rec:.3f} F1:{f1:.3f}", end="")
    print()
    return acc, pre, rec, f1

if __name__ == '__main__':
    dataset_name = 'ton' # or (ids18)
    model_name = 'mlp' # or (cnn, rescnn, lstm, Selfattention)
    model_type = 't' # or (s)

    fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{model_type}.csv')
    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_minmax_{model_type}.csv')
    fp_dataset = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_test_{model_type}.csv')
    fp_model = os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models', f'{dataset_name}_{model_name}_{model_type}.pth')

    main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_model)