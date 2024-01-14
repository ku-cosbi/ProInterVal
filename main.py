import torch
import data_prep
import random
from model import model as rep_model
from model import gnn
import pickle


def split_data(data_list, train_ratio, val_ratio=0.0):
    random.shuffle(data_list)
    train_size = int(len(data_list) * train_ratio)
    train_set = data_list[:train_size]
    val_size = int(len(data_list) * val_ratio)
    val_set = data_list[train_size:train_size+val_size]
    test_set = data_list[train_size+val_size:]

    return train_set, val_set, test_set


if __name__ == '__main__':
    data = data_prep.prepare_data()
    train_data, test_data = split_data(data, 0.8)

    num_epochs = 30
    learning_rate = 5e-4

    representation_model = rep_model.train(train_data, num_epochs, learning_rate)
    torch.save(representation_model, "../../models/representation_model.pt")
    deepinterface_data = data_prep.get_deepinterface_data()
    train, val, test = split_data(deepinterface_data, 0.8, 0.1)

    train_lst, val_lst, test_lst = [], [], []
    for i, t in enumerate(train):
        embedding = rep_model.get_interface_representation(representation_model, t[1], t[2])
        train_lst.append(embedding)

    for i, t in enumerate(val):
        embedding = rep_model.get_interface_representation(representation_model, t[1], t[2])
        val_lst.append(embedding)

    for i, t in enumerate(test):
        embedding = rep_model.get_interface_representation(representation_model, t[1], t[2])
        test_lst.append(embedding)

    gnn_model = gnn.train(train)
    torch.save(gnn_model, "../../models/gnn_model.pt")

    test_dct = dict()
    for i, t in enumerate(test):
        X, A = t[1], t[2]
        y = t[3]
        y_hat = gnn.predict(gnn_model, X, A)
        test_dct[t] = y_hat

    # save deepinterface test data
    with open('deepinterface_test.pkl', 'wb') as f:
        pickle.dump(test_dct, f)

