import pickle as pkl
import tqdm

with open("data/mydataset2.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in tqdm.tqdm(trans_seq.items()):
    for i in range(len(transactions)):
        if transactions[i][2] < 0:
            tmp = transactions[i][2]
            transactions[i][2] = -tmp
            transactions[i][3] = 1
        else:
            transactions[i][3] = 0
for address, transactions in tqdm.tqdm(trans_seq.items()):
    for j in range(len(transactions)):
        transactions[j] = [item for index, item in enumerate(transactions[j]) if index not in [0, 1, 4, 5, 6]]

with open("data/mydataset3.pkl", "wb") as f:
    pkl.dump(trans_seq, f)
print(1)
