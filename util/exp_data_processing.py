import pickle
import matplotlib.pyplot as plt

def visualize_single_dataline(single_data_line, label = None, color = None):
    if color is None:
        plt.plot(single_data_line, label = label)
    else:
        plt.plot(single_data_line, label = label, color = color)

def Exp1_F1_1_get_averaged_true_data_from_paths(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
    single_data_in_each_exp = []
    for single in data:
        temp = []
        for dataline in single[0]:
            temp.append(dataline[0])
        single_data_in_each_exp.append(temp)
    averaged_data = []
    for i in range(len(single_data_in_each_exp[0])):
        temp = []
        for j in range(len(single_data_in_each_exp)):
            temp.append(single_data_in_each_exp[j][i])
        averaged_data.append(sum(temp)/len(temp))
        
    single_data_in_each_exp = []
    for single in data:
        temp = []
        for dataline in single[1]:
            temp.append(dataline[0])
        single_data_in_each_exp.append(temp)
    averaged_pesudo_data = []
    for i in range(len(single_data_in_each_exp[0])):
        temp = []
        for j in range(len(single_data_in_each_exp)):
            temp.append(single_data_in_each_exp[j][i])
        averaged_pesudo_data.append(sum(temp)/len(temp))
    return averaged_data, averaged_pesudo_data

def Exp1_F1_3_get_full_true_data_from_paths(paths, layer):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
    ret = []
    for single in data:
        for sample in single[0][layer][2]:
            ret.append(sample)
    return ret

def get_ppls_from_paths(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
    ret = []
    for single in data:
        for sample in single:
            ret.append(sample)
    return ret

def Exp2_F2_get_averaged_hidden_cali_data_from_paths(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
    averaged = []
    for layer in range(len(data[0])):
        data_in_layer = []
        for single in data:
            data_in_layer.append(single[layer])
        averaged.append(sum(data_in_layer)/len(data_in_layer))
    return averaged, data

def Exp3_F1_get_averaged_data_from_paths(paths):
    data = []
    for path in paths:
        with open(path, 'rb') as f:
            data.append(pickle.load(f))
    averaged = []
    for layer in range(len(data[0])):
        data_in_layer = []
        for single in data:
            data_in_layer.append(single[layer][0])
        averaged.append(sum(data_in_layer)/len(data_in_layer))
    return averaged, data