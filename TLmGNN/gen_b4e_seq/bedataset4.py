import pickle as pkl
import tqdm

with open("data/bert4eth_trans3.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in tqdm.tqdm(trans_seq.items()):
    for i in range(len(transactions)):
        transactions[i] = transactions[i][2:]
        if transactions[i][1] == 'IN':
            transactions[i][1] = 0
        elif transactions[i][1] == 'OUT':
            transactions[i][1] = 1


with open("data/bert4eth_trans4.pkl", "wb") as f:
    pkl.dump(trans_seq, f)
print(1)