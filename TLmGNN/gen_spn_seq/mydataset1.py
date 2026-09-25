import pickle as pkl
from scipy.sparse import csr_matrix
import random

max_lenth = 100

with open("data/mydataset1.pkl", "rb") as f:
# with open("transaction_seq.pkl", "rb") as f:
    trans_seq = pkl.load(f)

with open("phisher_account.txt", "r") as f:
    phisher_addresses = {line.strip() for line in f}

address_to_idx = {}
idx_to_address = {}
undirect_trans_freq = {}
index = 0

for key, transactions in trans_seq.items():
    if len(transactions) < 1 or len(transactions) > max_lenth:
        continue
    if key not in address_to_idx:
        address_to_idx[key] = index
        idx_to_address[index] = key
        index += 1
    for transaction in transactions:
        vice_address = transaction[0]
        amount = transaction[2]
        if vice_address not in address_to_idx:
            address_to_idx[vice_address] = index
            idx_to_address[index] = vice_address
            index += 1
        pair = tuple(sorted([address_to_idx[key], address_to_idx[vice_address]]))
        if pair in undirect_trans_freq:
            undirect_trans_freq[pair] += 1
        else:
            undirect_trans_freq[pair] = 1

num_of_address = len(address_to_idx)
data, row_indices, col_indices = [], [], []
for (addr1, addr2), freq in undirect_trans_freq.items():
    row_indices.append(addr1)
    col_indices.append(addr2)
    data.append(freq)
adj = csr_matrix((data, (row_indices, col_indices)), shape=(num_of_address, num_of_address))

with open('data_train_spn/address_to_idx.pkl', 'wb') as f:
    pkl.dump(address_to_idx, f)

with open('data_train_spn/idx_to_address.pkl', 'wb') as f:
    pkl.dump(idx_to_address, f)

with open('data_train_spn/adj.pkl', 'wb') as f:
    pkl.dump(adj, f)

print(1)
