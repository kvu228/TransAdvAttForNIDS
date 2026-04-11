import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)
from utils.utils import CustomDataset, init_net, STORAGE_DIR
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader

def main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_output):
    dev = torch.device("cuda")
    lr = 0.001
    epoch = 10
    batch_size = 128

    dataset = CustomDataset(fp_dataset, fp_minmax, fp_fea)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = init_net(model_type, f'{model_name}_{model_type}')
    net.to(dev)

    criterion = CrossEntropyLoss()
    optimizar = Adam(net.parameters(), lr=lr, betas=(0.99, 0.99))

    for i in range(epoch):
        net.train()
        cur_iter = 0
        for data, labels in dataloader:
            cur_iter += len(data)
            data, labels = data.to(dev), labels.to(dev)

            optimizar.zero_grad()
            loss = criterion(net(data), labels)
            loss.backward()
            optimizar.step()
            print(f'\rEpoch:{i+1}/{epoch}, Progress:{cur_iter}/{len(dataset)}, Loss:{loss:.3f}', end='    ')
        print()
    torch.save(net.state_dict(), os.path.join(fp_output, f"{dataset_name}_{model_name}_{model_type}.pth"))

if __name__ == '__main__':
    dataset_name = 'ton' # or (ids18)
    model_name = 'mlp' # or (cnn, rescnn, lstm, Selfattention)
    model_type = 't' # or (s)

    fp_fea = os.path.join(STORAGE_DIR, 'dataset', f'fea_{model_type}.csv')
    fp_minmax = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_minmax_{model_type}.csv')
    fp_dataset = os.path.join(STORAGE_DIR, 'dataset', f'{dataset_name}_sam_train_{model_type}.csv')

    fp_output = os.path.join(STORAGE_DIR, 'custom', 'pre-trained_models')

    main(dataset_name, model_name, model_type, fp_fea, fp_minmax, fp_dataset, fp_output)