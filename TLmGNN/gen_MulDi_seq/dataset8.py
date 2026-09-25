import pickle
import random

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def select_and_shuffle_accounts(accounts):
    tag1_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 1]
    tag0_accounts = [account for account in accounts.items() if account[1][0]['tag'] == 0]

    double_tag1_count = random.sample(tag0_accounts, 2 * len(tag1_accounts))

    selected_accounts = tag1_accounts + double_tag1_count
    random.shuffle(selected_accounts)

    return dict(selected_accounts)

accounts_data = load_data('./data/transactions7.pkl')

shuffled_accounts_data = select_and_shuffle_accounts(accounts_data)

save_data(shuffled_accounts_data, 'transactions8.pkl')


