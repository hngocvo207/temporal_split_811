import pickle
import tqdm

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

accounts_data = load_data('./data/transactions3.pkl')

add_n_grams(accounts_data)

save_data(accounts_data, './data/transactions4.pkl')

