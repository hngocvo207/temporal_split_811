import pickle
from sklearn.model_selection import train_test_split

with open('./data/transactions10.pkl', 'rb') as file:
    transactions_dict = pickle.load(file)

with open('Dataset/address_to_idx.pkl', 'rb') as file:
    address_to_idx = pickle.load(file)

transactions = []
for key, value_list in transactions_dict.items():
    for value in value_list:
        index = address_to_idx[key]
        transactions.append(f"{index} {value}")

train_size = 0.8
validation_size = 0.1
test_size = 0.1

train_data, temp_data = train_test_split(transactions, train_size=train_size, random_state=42)

validation_data, test_data = train_test_split(temp_data, test_size=test_size/(test_size + validation_size), random_state=42)

def save_to_tsv_train_dev(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tlabel\tsentence\n")
        for line in data:
            index, rest = line.split(' ', 1)
            tag, sentence = rest.split(' ', 1)
            file.write(f"{index}\t{tag}\t{sentence}\n")

def save_to_tsv_test(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write("index\tlabel\tsentence\n")
        for idx, line in enumerate(data):
            index, rest = line.split(' ', 1)
            tag, sentence = rest.split(' ', 1)
            file.write(f"{index}\t{tag}\t{sentence}\n")

save_to_tsv_train_dev(train_data, 'trans_sentence/train.tsv')
save_to_tsv_train_dev(validation_data, 'trans_sentence/dev.tsv')
save_to_tsv_test(test_data, 'trans_sentence/test.tsv')

print("Files saved: train.tsv, dev.tsv, test.tsv")
