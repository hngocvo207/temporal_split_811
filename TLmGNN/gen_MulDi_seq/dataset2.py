import pickle

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def process_transactions(transactions):
    accounts = {}

    for tx in transactions:
        from_address = tx['from_address']
        if from_address not in accounts:
            accounts[from_address] = []
        accounts[from_address].append({**tx, 'in_out': 1})

        to_address = tx['to_address']
        if to_address not in accounts:
            accounts[to_address] = []
        accounts[to_address].append({**tx, 'in_out': 0})

    return accounts

transactions = load_data('./data/transactions1.pkl')

processed_data = process_transactions(transactions)

save_data(processed_data, './data/transactions2.pkl')


