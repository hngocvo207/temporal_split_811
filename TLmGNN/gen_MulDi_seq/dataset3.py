import pickle

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def sort_transactions_by_timestamp(accounts):
    sorted_accounts = {}
    for address, transactions in accounts.items():
        sorted_accounts[address] = sorted(transactions, key=lambda x: x['timestamp'])
    return sorted_accounts

accounts_data = load_data('./data/transactions2.pkl')

sorted_accounts_data = sort_transactions_by_timestamp(accounts_data)

# 保存数据
save_data(sorted_accounts_data, './data/transactions3.pkl')

