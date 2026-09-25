import pickle

def load_data(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_data(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def convert_transactions_to_text(accounts):
    for address, transactions in accounts.items():
        for idx, transaction in enumerate(transactions):
            tag = transaction['tag']
            transaction_descriptions = []
            for sub_transaction in transaction['transactions']:
                description = ' '.join([f"{key}: {sub_transaction[key]}" for key in sub_transaction])
                transaction_descriptions.append(description)
            transactions[idx] = f"{tag} {'  '.join(transaction_descriptions)}."

accounts_data = load_data('./data/transactions9.pkl')

convert_transactions_to_text(accounts_data)

save_data(accounts_data, './data/transactions10.pkl')

