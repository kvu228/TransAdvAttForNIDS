import os, sys, glob
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

from utils.utils import STORAGE_DIR

import dpkt, socket, torch, numpy as np, pandas as pd
from collections import defaultdict, deque
from utils.tantra import TantraLSTM

def ip_to_str(b: bytes) -> str: return socket.inet_ntoa(b)
def flow_id(src_ip,dst_ip,sport,dport,proto):
    return (f"{dst_ip}-{src_ip}-{dport}-{sport}-{proto}"
            if dst_ip < src_ip else
            f"{src_ip}-{dst_ip}-{sport}-{dport}-{proto}")

def norm_delay(x): return (x - d_min) / (d_max - d_min)
def denorm_delay(x): return x * (d_max - d_min) + d_min
def norm_size(x):  return (x - s_min) / (s_max - s_min)

def get_onebenflow():
    fp = os.path.join(STORAGE_DIR, 'tantra', '1000_ben_flow.csv')
    df = pd.read_csv(fp, header=0, index_col=None)
    df_oneflow = df.sample(1, replace=False, axis=0)
    
    # normalization
    ts_cols = sorted([c for c in df.columns if c.startswith("ts_-")],
                        key=lambda s: int(s.split("_")[-1]))        # ts_-150 … ts_-1
    sz_cols = sorted([c for c in df.columns if c.startswith("sz_-")],
                    key=lambda s: int(s.split("_")[-1]))        # sz_-150 … sz_-1

    # ---------------- calculate Δt ----------------
    ts_values = df_oneflow[ts_cols].values      # (N,150)
    delta_ts  = np.zeros_like(ts_values)
    delta_ts[:, 1:] = ts_values[:, 1:] - ts_values[:, :-1]  # Δt[0]=0

    delta_ts = norm_delay(delta_ts)
    np_sz = df_oneflow[sz_cols]
    np_sz = norm_size(np_sz)

    df_oneflow[ts_cols] = delta_ts
    df_oneflow[sz_cols] = np_sz
    
    lst_one_ben_flow = []
    for i in range(1, 300, 2):
        lst_one_ben_flow.append((df_oneflow.iloc[0, i], df_oneflow.iloc[0, i+1]))
    return lst_one_ben_flow

def parser_pcap(fp_pcap_in):
    # -------- 2. parser PCAP & build index ---------------------------------------------
    pkt_list = []                       # (orig_idx, ts, buf, flow_id, size, src_ip)
    flows = defaultdict(list)           # flow_id → list[idx]
    try:
        with open(fp_pcap_in, "rb") as f:
            for idx, (ts, buf) in enumerate(dpkt.pcap.Reader(f)):
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP): continue
                ip = eth.data ; proto = ip.p
                if proto==dpkt.ip.IP_PROTO_TCP and isinstance(ip.data,dpkt.tcp.TCP): l4=ip.data
                elif proto==dpkt.ip.IP_PROTO_UDP and isinstance(ip.data,dpkt.udp.UDP): l4=ip.data
                else: continue
                fid = flow_id(ip_to_str(ip.src), ip_to_str(ip.dst), l4.sport, l4.dport, proto)
                pkt_list.append([idx, ts, buf, fid, len(buf), ip_to_str(ip.src)])
                flows[fid].append(idx)
    except dpkt.UnpackError as e:
            pass
    return pkt_list, flows

def rewrite_ts(pkt_list, flows, ws, dev, model, lst_one_ben_flow):
    # -------- 3. generate new timestamp ----------------------------------------
    new_ts = np.zeros(len(pkt_list), dtype=float)

    c1 = 0
    for fid, indices in flows.items():
        # sort by timestamp
        indices.sort(key=lambda i: pkt_list[i][1])
        window = deque(maxlen=ws)

        first_idx = indices[0]
        size0 = pkt_list[first_idx][4]

        window.extend(lst_one_ben_flow)
        window.append((0.0, norm_size(size0))) # delay of first pkt is 0

        last_abs_ts = pkt_list[first_idx][1]
        new_ts[first_idx] = last_abs_ts

        # record forward pkt
        att_ip = pkt_list[first_idx][5]

        # process the remaining pkts
        last_idx = first_idx
        for k in range(1, len(indices)):
            
            idx = indices[k]
            size_k = pkt_list[idx][4]
            cur_pkt_ip = pkt_list[idx][5]

            # (1, WS+1, 2)
            feat_hist = np.array(window, dtype=np.float32)          # (150,2)
            cur_feat  = np.array([[0.0, norm_size(size_k)]],dtype=np.float32)
            inp = np.concatenate([feat_hist, cur_feat], axis=0)     # (151,2)
            inp_t = torch.from_numpy(inp).unsqueeze(0).to(dev)   # (1,151,2)

            if cur_pkt_ip == att_ip:
                with torch.no_grad():
                    delay_pred_norm = model(inp_t).item()               # scalar 0-1
            else:
                delay_pred_norm = -1.

            if delay_pred_norm > 0:
                c1 += 1
                delay_pred = denorm_delay(delay_pred_norm) 
            else:
                delay_pred = pkt_list[idx][1] - pkt_list[last_idx][1]

            # calculate new timestamp
            last_abs_ts += delay_pred
            new_ts[idx] = last_abs_ts

            # update window
            window.append((norm_delay(delay_pred), norm_size(size_k)))
            last_idx = idx
    print(c1)
    return new_ts

def wirte_pcap(fp_pcap_out, pkt_list, new_ts):
    # -------- 4. write new PCAP ----------------------------------------------------
    try:
        with open(fp_pcap_out, "wb") as f_out:
            writer = dpkt.pcap.Writer(f_out)
            for i, (orig_idx, _, buf, _, _, _) in enumerate(pkt_list):
                writer.writepkt(buf, ts=new_ts[orig_idx])
    except dpkt.UnpackError as e:
            pass


if __name__ == '__main__':

    fp_model = os.path.join(STORAGE_DIR, 'tantra', 'tantra_lstm.pth')
    ws = 150
    dev = torch.device("cuda")
    fp_minmax = os.path.join(STORAGE_DIR, 'tantra', 'tantra_minmax.csv')

    # -------- 1. load model and scaler -----------------------------------------------
    model = TantraLSTM().to(dev)
    model.load_state_dict(torch.load(fp_model, map_location=dev, weights_only=True))
    model.eval()

    df_minmax = pd.read_csv(fp_minmax, header=0, index_col=None)
    d_min, d_max = df_minmax.iloc[0, 0], df_minmax.iloc[1, 0]
    s_min, s_max = df_minmax.iloc[0, 1], df_minmax.iloc[1, 1]

    # random pick one benign traffic for modify partial pkts in window
    lst_one_ben_flow = get_onebenflow()

    lst_fp_allpcap = glob.glob(os.path.join(STORAGE_DIR, 'att_pcap', '*'))
    for i, fp_onepcap in enumerate(lst_fp_allpcap):
        print(f"Reading {os.path.basename(fp_onepcap)}({i})")

        fp_pcap_out = os.path.join(STORAGE_DIR, 'tantra', 'adv_pcap', os.path.basename(fp_onepcap))

        # -------- 2. parser PCAP & build index ---------------------------------------------
        pkt_list, flows = parser_pcap(fp_onepcap)

        # -------- 3. generate new timestamp ----------------------------------------
        print('Cal ts...')
        new_ts = rewrite_ts(pkt_list, flows, ws, dev, model, lst_one_ben_flow)

        # -------- 4. write new PCAP ----------------------------------------------------
        print('Write pcap...')
        wirte_pcap(fp_pcap_out, pkt_list, new_ts)
