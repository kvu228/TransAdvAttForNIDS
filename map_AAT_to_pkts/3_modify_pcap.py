import os, sys
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_dir)

from utils.utils import STORAGE_DIR

import dpkt
import pandas as pd
import socket
import struct
from collections import defaultdict
import os, glob

# ───────────────────────────────────────────────────────────────
# Helper functions
# ───────────────────────────────────────────────────────────────
def inet_to_str(inet):
    """bytes -> dotted string"""
    return socket.inet_ntoa(inet)

def get_flow_id(src_ip, dst_ip, sport, dport, proto):
    """Flow-ID rule described by the user"""
    if dst_ip < src_ip:
        return f"{dst_ip}-{src_ip}-{dport}-{sport}-{proto}"
    return f"{src_ip}-{dst_ip}-{sport}-{dport}-{proto}"

def fix_checksums(ip):
    """Re-compute IP and L4 checksums in-place (IPv4 only)."""
    # Reset
    ip.sum = 0
    ip.len = len(ip)             # update total length

    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        l4.sum = 0
        pseudo = struct.pack("!4s4sBBH", ip.src, ip.dst, 0, ip.p, len(l4))
        l4.sum = dpkt.in_cksum(pseudo + bytes(l4))
    elif isinstance(l4, dpkt.udp.UDP):
        l4.sum = 0
        l4.ulen = len(l4)
        pseudo = struct.pack("!4s4sBBH", ip.src, ip.dst, 0, ip.p, l4.ulen)
        l4.sum = dpkt.in_cksum(pseudo + bytes(l4))

    ip.sum = dpkt.in_cksum(ip.pack_hdr() + bytes(ip.data))

def pad_packet(ip, need_len):
    """Increase payload length to `need_len` by zero-padding."""
    l4 = ip.data
    pay = bytearray(l4.data)
    # extra = need_len - len(pay)
    pay.extend(b"\x00" * need_len)
    l4.data = bytes(pay)

    if isinstance(l4, dpkt.udp.UDP):
        l4.ulen = len(l4)  # update UDP len
    fix_checksums(ip)

def truncate_packet(ip, need_len):
    """Shorten payload length to `need_len`."""
    l4 = ip.data
    l4.data = l4.data[:max(1, len(l4.data) - need_len)]
    if isinstance(l4, dpkt.udp.UDP):
        l4.ulen = len(l4)
    fix_checksums(ip)

# ───────────────────────────────────────────────────────────────
# Load CSV: flow_id ↦ desired extrema (dict)
# ───────────────────────────────────────────────────────────────
def load_targets(csv_path):
    df = pd.read_csv(csv_path)

    tgt = {}
    for _, row in df.iterrows():
        fid = row["flow_id"]
        tgt[fid] = {
            "src_ip": row["Src IP"],
            "pkt_len_max": int(row["Fwd Pkt Len Max"]),
            "pkt_len_min": int(row["Fwd Pkt Len Min"]),
            "iat_max":  float(row["Fwd IAT Max"]),
            "iat_min":  float(row["Fwd IAT Min"]),
        }
    return tgt

# ───────────────────────────────────────────────────────────────
# Main routine
# ───────────────────────────────────────────────────────────────
def generate_adv_pcap(in_pcap, targets, out_pcap):

    

    # Buffers
    all_packets = []                      # (ts, raw)
    flow_pkts = defaultdict(list)         # fid -> list[(idx, ts, ip)]

    # 1. parser pcap
    with open(in_pcap, "rb") as f:
        reader = dpkt.pcap.Reader(f)
        for pkt_idx, (ts, raw) in enumerate(reader):
            eth = dpkt.ethernet.Ethernet(raw)
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                all_packets.append([ts, raw, None])  # keep untouched
                continue

            proto = ip.p
            if proto not in (dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP):
                all_packets.append([ts, raw, None])
                continue

            l4 = ip.data
            fid = get_flow_id(
                inet_to_str(ip.src),
                inet_to_str(ip.dst),
                l4.sport,
                l4.dport,
                proto,
            )
            all_packets.append([ts, raw, fid])

            if fid in targets:   # we only store flows we’ll touch
                flow_pkts[fid].append([pkt_idx, ts, eth])

    
    # 2. modify forward traffic
    for fid, packets in flow_pkts.items():
        t = targets[fid]
        src_ip_fwd = t["src_ip"]

        # --- 2.1 filter pkts ---
        fwd = [[i, ts, eth] for i, ts, eth in packets
               if inet_to_str(eth.data.src) == src_ip_fwd]
        if len(fwd) < 2:
            continue

        # --- 2.2 process packet length extrema ---
        lengths = [len(p[2].data.data.data) if isinstance(p[2].data, dpkt.udp.UDP)
                   else len(p[2].data.data) for p in fwd]

        # current extrema
        cur_max_len = max(lengths)
        cur_min_len = min(lengths)

        # increase the maximun
        if t["pkt_len_max"] > 0:
            idx = lengths.index(cur_max_len)
            pkt_idx, ts, eth = fwd[idx]
            pad_packet(eth.data, t["pkt_len_max"])
            all_packets[pkt_idx][1] = bytes(eth)   # overwrite raw

        # decrease the minimun
        if t["pkt_len_min"] < 0:
            idx = lengths.index(cur_min_len)
            pkt_idx, ts, eth = fwd[idx]
            truncate_packet(eth.data, t["pkt_len_min"])
            all_packets[pkt_idx][1] = bytes(eth)

        # --- 2.3 precess IAT extrema ---
        # sort by timestamp
        fwd.sort(key=lambda x: x[1])
        times = [ts for _, ts, _ in fwd]
        # print(times)
        # break

        # list of IAT
        iats = [times[i] - times[i-1] for i in range(1, len(times))]
        cur_max_iat = max(iats)
        cur_min_iat = min(iats)
        # print("-1-", cur_max_iat, cur_min_iat)
        # print("-2-", t["iat_max"], t["iat_min"])

        # increase the maximun IAT
        if t["iat_max"] > 0:
            idx = iats.index(cur_max_iat) + 1
            delta = t["iat_max"] * 1e-6
            # move the pkts after idx by delta
            for j in range(idx, len(fwd)):
                pkt_idx, ts, eth = fwd[j]
                new_ts = ts + delta
                fwd[j][1] = new_ts
                all_packets[pkt_idx][0] = new_ts

        # decrease the minimun IAT
        if t["iat_min"] < 0:
            idx = iats.index(cur_min_iat) + 1
            delta = abs(t["iat_min"]) * 1e-6
            # cannot earlier than last pkt
            prev_ts = fwd[idx-1][1]
            new_ts_candidate = max(prev_ts, fwd[idx][1] - delta)
            shift = new_ts_candidate - fwd[idx][1]
            for j in range(idx, len(fwd)):
                pkt_idx, ts, eth = fwd[j]
                new_ts = ts + shift
                fwd[j][1] = new_ts
                all_packets[pkt_idx][0] = new_ts

    # 3. write new pcap
    print('Writting pkts...')
    with open(out_pcap, "wb") as fout:
        writer = dpkt.pcap.Writer(fout)
        for ts, raw, _ in all_packets:
            writer.writepkt(raw, ts)

# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    csv_path = os.path.join(project_root_dir, 'output', 'diff.csv')
    targets = load_targets(csv_path)

    lst_fp_allpcap = glob.glob(os.path.join(STORAGE_DIR, 'att_pcap', '*'))
    for i, fp_onepcap in enumerate(lst_fp_allpcap):
        print(f'Reading {os.path.basename(fp_onepcap)}({i})')

        out_pcap = os.path.join(STORAGE_DIR, 'adv_pcap', os.path.basename(fp_onepcap))
        generate_adv_pcap(fp_onepcap, targets, out_pcap)

