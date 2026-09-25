import pickle as pkl
import tqdm

with open("data/bert4eth_trans2.pkl", "rb") as f:
    trans_seq = pkl.load(f)

for address, transactions in trans_seq.items():
    # Sort transactions by timestamp (the second element of each transaction)
    sorted_transactions = sorted(transactions, key=lambda x: x[1])
    # Update the transactions for the current address with the sorted list
    trans_seq[address] = sorted_transactions

with open("data/bert4eth_trans3.pkl", "wb") as f:
    pkl.dump(trans_seq, f)

print(1)
