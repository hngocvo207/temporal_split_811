import pickle as pkl
import tqdm

with open("data/mydataset1.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in trans_seq.items():
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions


with open("data/mydataset2.pkl", "wb") as f:
    pkl.dump(trans_seq, f)

print(1)
