import os
import csv
import pickle

maxvalue = 0
maxfrom = ""
maxto = ""
minvalue = 0
minfrom = ""
minto = ""
maxtime = 0

def bucketization(value, ioflag):
    if value == 0:
        value = 1
    elif 0 < value <= 0.000591:
        value = 2
    elif 0 < value <= 0.006195:
        value = 3
    elif 0 < value <= 0.021255:
        value = 4
    elif 0 < value <= 0.050161:
        value = 5
    elif 0 < value <= 0.100120:
        value = 6
    elif 0 < value <= 0.208727:
        value = 7
    elif 0 < value <= 0.508961:
        value = 8
    elif 0 < value <= 1.360574:
        value = 9
    elif 0 < value <= 6.500000:
        value = 10
    elif 0 < value <= 143791.433950:
        value = 11
    else:
        value = 12
    # if ioflag == -1:
    #     value = ioflag * value
    return value

def csv_tran2dict_tran(tran_seq, main_address, csv_file):
    global processed_files, maxvalue, minvalue, maxfrom, maxto, minfrom, minto
    if main_address in processed_files and processed_files[main_address]:
        return tran_seq
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            timestamp = int(row[3])
            sender = row[4]
            receiver = row[5]
            value = float(row[6])
            gas = int(row[7])
            gasprice = int(row[8])
            gasused = int(row[9])
            is_error = row[12]
            io_flag = 0
            if main_address == sender:
                io_flag = -1
                vice_address = receiver
            else:
                io_flag = 1
                vice_address = sender

            value = io_flag * value
            # value = bucketization(value, io_flag)
            if value > 0 and value > maxvalue:
                maxvalue = value
                maxfrom = sender
                maxto = receiver
            if value < 0 and value < minvalue:
                minvalue = value
                minfrom = sender
                minto = receiver

            if value != 0 and is_error == '0':
                if main_address not in tran_seq:
                    tran_seq[main_address] = []
                tran_seq[main_address].append([vice_address, timestamp, value, gas, gasprice, gasused])
    processed_files[main_address] = True
    return tran_seq


def update_timestamp_average(seq_tmp, i, windows):
    global maxtime
    timestamps = []
    half_window = (windows - 1) // 2
    errorflag = 0

    for j in range(i - half_window, i + half_window + 1):
        if j < 0 and j - 1 < 0:
            timestamps.append(0)
        elif j == 0:
            timestamps.append(1)
        elif j > 0 and j < len(seq_tmp):
            time_diff = seq_tmp[j][1] - seq_tmp[j - 1][1]
            if time_diff < 0:
                errorflag = 1
                print("error!!!!!!" + '***' + str(seq_tmp[j][1]) + '***' + str(seq_tmp[j - 1][1]))
            timestamps.append(time_diff)
        elif j >= len(seq_tmp):
            timestamps.append(0)
    avg_time_diff = sum(timestamps) / len(timestamps) if timestamps else 0

    avg_time_diff = avg_time_diff // 3600
    if avg_time_diff > maxtime:
        maxtime = avg_time_diff
    seq_tmp[i].append(avg_time_diff)

    return seq_tmp, errorflag

csv.field_size_limit(1000000)

data_path = "/home/amax/Desktop/etherscan_download_data/all_len100"
# data_path = "data"

transaction_seq = {}
processed_files = {}

for first_account_folder_name in os.listdir(data_path):
    first_account_folder_dir = os.path.join(data_path, first_account_folder_name)
    if os.path.isdir(first_account_folder_dir) and first_account_folder_name.endswith("_data"):
        first_account_name = first_account_folder_name.replace("_data", "")
        csv_file_path = os.path.join(first_account_folder_dir, f"{first_account_name}.csv")

        transaction_seq = csv_tran2dict_tran(transaction_seq, first_account_name, csv_file_path)

        next_account_transaction_folder_name = os.path.join(first_account_folder_dir,
                                                            first_account_name)

        for csv_file in os.listdir(next_account_transaction_folder_name):
            file_path = os.path.join(next_account_transaction_folder_name, csv_file)
            if csv_file.endswith(".csv") and not csv_file.endswith("_neighbor.csv"):
                transaction_seq = csv_tran2dict_tran(transaction_seq, csv_file.replace(".csv", ""), file_path)

for trans in transaction_seq.keys():
    seq_tmp = transaction_seq[trans]
    for i in range(len(seq_tmp) - 1, -1, -1):
        windows = 5
        seq_tmp, flag = update_timestamp_average(seq_tmp, i, windows)
        if flag == 1: print(trans)
    transaction_seq[trans] = seq_tmp

# with open("data/transaction_seq.pkl", 'wb') as file:
# with open("test_data/transaction_seq.pkl", 'wb') as file:
#     pickle.dump(transaction_seq, file)

with open("all_len100/transaction_seq.pkl", 'wb') as file:
    pickle.dump(transaction_seq, file)

print(1)
