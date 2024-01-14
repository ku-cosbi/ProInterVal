import os
import torch
import data_prep
from model import model as rep_model
from model import gnn
import pickle
import argparse
import requests
from urllib.parse import urlparse


def load_file_from_url(url):
    parts = urlparse(url)
    file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(os.getcwd(), file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file)
    return cached_file

parser = argparse.ArgumentParser()
parser.add_argument("mode", help="Train or test. Train model from scratch or test pretrained model using provided dataset")
parser.add_argument("path", help="Path to your database")
parser.add_argument("model", help="RL for representation learning model, GNN for interface validation model.")
parser.parse_args()

representation_model_path = "https://huggingface.co/cosbi-ku/ProInterVal/resolve/main/representation_model.pt"
validation_model_path = "https://huggingface.co/cosbi-ku/ProInterVal/resolve/main/gnn_model.pt"


if __name__ == '__main__':
    if parser.path:
        if parser.model == "RL":
            if parser.mode == "train":
                train_data = data_prep.prepare_data(parser.path)
                num_epochs = 30
                learning_rate = 5e-4
                representation_model = rep_model.train(train_data, num_epochs, learning_rate)
                torch.save(representation_model, "representation_model.pt")
                print("Trained model is saved as representation_model.pt into your current directory.")
            elif parser.mode == "test":
                test_data = data_prep.prepare_data(parser.path)

                # Load representation learning model
                model = rep_model.RepresentationLearningModel()

                model_path = load_file_from_url(representation_model_path)
                model.load_state_dict(torch.load(model_path))
                test_dct = {}
                for i, t in enumerate(test_data):
                    embedding = rep_model.get_interface_representation(model, t[1], t[2])
                    test_dct[t[0]] = embedding

                with open('learned_representations.pkl', 'wb') as f:
                    pickle.dump(test_dct, f)

                print("Learned representations are saved as learned_representations.pkl into your current directory.")
            else:
                print(f"Unknown mode: {parser.mode}. Available modes: train, test")
        elif parser.model == "GNN":
            if parser.mode == "train":
                train = data_prep.get_deepinterface_data(parser.path)
                gnn_model = gnn.train(train)
                torch.save(gnn_model, "gnn_model.pt")
                print("Trained model is saved as gnn_model.pt into your current directory.")
            elif parser.mode == "test":
                test = data_prep.get_deepinterface_data(parser.path)
                
                num_features = len(test[0])
                hidden_size = 512
                num_classes = 2
                learning_rate = 1.0e-3
                num_epochs = 20
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load model
                model = gnn.GNNModel(num_features, hidden_size, num_classes).to(device)
                model_path = load_file_from_url(validation_model_path)
                model.load_state_dict(torch.load(model_path))

                result_file = open("interface_val_results.txt", "w")

                for i, t in enumerate(test):
                    X, A = t[1], t[2]
                    y_hat = gnn.predict(model, X, A)
                    result_file.write(f"{t[0]}: {y_hat}")

                result_file.close()
                print("Results are saved as interface_val_results.txt into your current directory.")
            else:
                print(f"Unknown mode: {parser.mode}. Available modes: train, test")
        else:
            print(f"Unknown model: {parser.model}. Available models: RL, GNN")
    else:
        print("You need to provide a path to your dataset directory.")
