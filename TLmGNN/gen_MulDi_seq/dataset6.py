import pickle
import random
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def shuffle_transactions(accounts):
    for address in tqdm.tqdm(accounts.keys()):
        random.shuffle(accounts[address])

accounts_data = load_data('./data/transactions5.pkl')

shuffle_transactions(accounts_data)

save_data(accounts_data, './data/transactions6.pkl')

