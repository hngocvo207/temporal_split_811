import pickle

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# 保存数据到文件
def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def remove_tag_from_transactions(accounts):
    for address, transactions in accounts.items():
        for transaction in transactions:
            for sub_transaction in transaction['transactions']:
                if 'tag' in sub_transaction:
                    del sub_transaction['tag']

accounts_data = load_data('./data/transactions8.pkl')

remove_tag_from_transactions(accounts_data)

save_data(accounts_data, './data/transactions9.pkl')

