import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def remove_fields(accounts, fields):
    for address in tqdm.tqdm(accounts.keys()):
        for transaction in accounts[address]:
            for field in fields:
                if field in transaction:
                    del transaction[field]

accounts_data = load_data('./data/transactions4.pkl')

fields_to_remove = ['from_address', 'to_address', 'timestamp']

remove_fields(accounts_data, fields_to_remove)

save_data(accounts_data, './data/transactions5.pkl')

