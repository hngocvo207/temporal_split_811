from tqdm import tqdm
import pickle
def read_pkl(pkl_file):
    print(f'Reading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pkl(data, pkl_file):
    print(f'Saving data to {pkl_file}...')
    with open(pkl_file, 'wb') as file:
        pickle.dump(data, file)
def load_and_print_pkl(pkl_file):
    print(f'Loading {pkl_file}...')
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)

    for i, transaction in enumerate(data):
        if i < 10:
            print(transaction)
        else:
            break
def extract_transactions(G):
    transactions = []
    for from_address, to_address, key, tnx_info in tqdm(G.edges(keys=True, data=True),desc=f'accounts_data_generate'):
        amount = tnx_info['amount']
        block_timestamp = int(tnx_info['timestamp'])
        tag = G.nodes[from_address]['isp']
        transaction = {
            'tag': tag,
            'from_address': from_address,
            'to_address': to_address,
            'amount': amount,
            'timestamp': block_timestamp,
        }
        transactions.append(transaction)
    return transactions

def data_generate():
    graph_file = './data/MulDiGraph.pkl'
    out_file = './data/transactions1.pkl'

    graph = read_pkl(graph_file)
    transactions = extract_transactions(graph)
    save_pkl(transactions, out_file)

if __name__ == '__main__':
    data_generate()
    pkl_file = 'data/transactions1.pkl'
    load_and_print_pkl(pkl_file)

