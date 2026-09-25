import pickle as pkl
from scipy.sparse import csr_matrix
from tqdm import tqdm

with open("transactions1.pkl", "rb") as f:
    trans_seq = pkl.load(f)
with open("Dataset/index_to_guid", "rb") as f:
    index_to_guid = pkl.load(f)

address_to_idx = {}
idx_to_address = {}
undirect_trans_freq = {}
index = 0

for transaction in tqdm(trans_seq):
    tag = transaction['tag']
    from_address = transaction['from_address']
    to_address = transaction['to_address']
    amount = transaction['amount']
    block_timestamp = transaction['timestamp']
    if from_address not in address_to_idx:
        address_to_idx[from_address] = index
        idx_to_address[index] = from_address
        index += 1

    if to_address not in address_to_idx:
        address_to_idx[to_address] = index
        idx_to_address[index] = to_address
        index += 1
    pair = tuple(sorted([address_to_idx[from_address], address_to_idx[to_address]]))
    if pair in undirect_trans_freq:
        undirect_trans_freq[pair] += 1
    else:
        undirect_trans_freq[pair] = 1

num_of_address = len(address_to_idx)
data, row_indices, col_indices = [], [], []
for (addr1, addr2), freq in tqdm(undirect_trans_freq.items()):
    if addr1 in index_to_guid and addr2 in index_to_guid:
        row_indices.append(addr1)
        col_indices.append(addr2)
        data.append(freq)

adj = csr_matrix((data, (row_indices, col_indices)), shape=(len(index_to_guid), len(index_to_guid)))

with open('Dataset/mini_adj.pkl', 'wb') as f:
    pkl.dump(adj, f)

with open('Dataset/address_to_idx.pkl', 'wb') as f:
    pkl.dump(address_to_idx, f)
print(1)
