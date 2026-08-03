"""
Sửa nhãn trong transactions7.pkl dựa trên ground truth từ MulDiGraph.

Pipeline gốc (dataset2.py + dataset6.py + dataset7.py) gán nhãn account
dựa trên tag của giao dịch ĐẦU TIÊN sau shuffle ngẫu nhiên. Điều này gây ra:
  - ~763 fraud accounts (isp=1) bị nhãn 0 (normal) vì giao dịch đầu
    tiên sau shuffle là "nhận tiền từ normal sender" → tag = 0
  - 1,505 normal accounts (isp=0) được nhãn 1 (fraud) vì giao dịch đầu
    tiên sau shuffle là "nhận tiền từ fraud sender" → tag = 1

Paper Dynamic Feature phát biểu:
  "even if only one transaction in the account is related to fraud,
   the account itself may be potentially risky."

Script này gán nhãn đúng theo ý định của paper:
  confirmed_fraud  : isp=1 trong MulDiGraph                           → tag = 1
  potential_fraud  : isp=0 VÀ có ít nhất 1 cạnh nhận từ node isp=1   → tag = 1
  pure_normal      : isp=0 VÀ không nhận từ bất kỳ node isp=1 nào    → tag = 0

Output: transactions7_corrected.pkl
  Cùng cấu trúc với transactions7.pkl nhưng trường 'tag' được đặt
  đúng theo phân loại trên.

Chạy một lần duy nhất, trước khi chạy vòng lặp ratio.
"""
import os
import pickle

MULDIGRAPH_PATH  = 'MulDiGraph.pkl'
TRANSACTIONS_IN  = 'transactions7.pkl'
TRANSACTIONS_OUT = 'transactions7_corrected.pkl'


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def main():
    if os.path.exists(TRANSACTIONS_OUT):
        print(f"{TRANSACTIONS_OUT} đã tồn tại, bỏ qua.")
        return

    # 1. Load MulDiGraph — nodes là address strings, attr 'isp' là ground truth
    print(f"Đang load {MULDIGRAPH_PATH} ...")
    G = load(MULDIGRAPH_PATH)
    print(f"  {G.number_of_nodes():,} nodes  {G.number_of_edges():,} edges")

    # 2. Xác định confirmed fraud (isp=1)
    fraud_addrs = {
        addr for addr in G.nodes()
        if G.nodes[addr].get('isp', 0) == 1
    }
    print(f"  Confirmed fraud (isp=1): {len(fraud_addrs):,}")

    # 3. Xác định potential fraud: isp=0, nhận tiền từ ≥1 node có isp=1
    #    Duyệt qua các fraud node → tìm các successor (node nhận tiền từ họ)
    potential_addrs: set = set()
    for fa in fraud_addrs:
        for succ in G.successors(fa):
            if G.nodes[succ].get('isp', 0) == 0:
                potential_addrs.add(succ)
    potential_addrs -= fraud_addrs   # đảm bảo không chồng lặp
    print(f"  Potential fraud (isp=0, received from fraud): {len(potential_addrs):,}")
    print(f"  Total positive class: {len(fraud_addrs) + len(potential_addrs):,}")

    del G   # giải phóng bộ nhớ

    # 4. Load transactions7.pkl và cập nhật tag
    print(f"\nĐang load {TRANSACTIONS_IN} ...")
    accounts = load(TRANSACTIONS_IN)
    print(f"  {len(accounts):,} accounts")

    n_confirmed = n_potential = n_normal = n_unknown = 0
    for addr, data_list in accounts.items():
        account = data_list[0]   # merged entry từ dataset7.py
        if addr in fraud_addrs:
            account['tag'] = 1
            n_confirmed += 1
        elif addr in potential_addrs:
            account['tag'] = 1
            n_potential += 1
        else:
            account['tag'] = 0
            n_normal += 1

    print(f"\nKết quả sau khi sửa nhãn:")
    print(f"  confirmed fraud  (isp=1)        : {n_confirmed:,}")
    print(f"  potential fraud  (isp=0→tag=1)  : {n_potential:,}")
    print(f"  pure normal      (isp=0→tag=0)  : {n_normal:,}")

    total_pos = n_confirmed + n_potential
    total = n_confirmed + n_potential + n_normal
    print(f"  Total positive                  : {total_pos:,} ({100*total_pos/total:.1f}%)")
    print(f"  Total accounts                  : {total:,}")

    # 5. Lưu
    save(accounts, TRANSACTIONS_OUT)
    print(f"\nĐã lưu: {TRANSACTIONS_OUT}")


if __name__ == '__main__':
    main()
