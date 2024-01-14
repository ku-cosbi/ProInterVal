import os
import torch
import data_prep
from model.model import *
from model.gnn import *
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

                input_size = 30  # Input size for nodes and edges in GCL Autoencoder
                latent_size = 128  # Size of the latent space
                view_size = 64  # View size for the MultiviewContrastiveLayer
                transformer_input_size = 64  # Input size for the Transformer
                hidden_size = 128  # Hidden size for the Transformer
                num_layers = 4  # Number of Transformer layers
                num_heads = 8  # Number of attention heads
                dropout = 0.1  # Dropout rate
                lambda0 = 0.2  # Weight for the node reconstruction loss
                lambda1 = 0.3  # Weight for the KL divergence loss
                lambda2 = 0.3  # Weight for the MSE loss
                lambda3 = 0.2  # Weight for the edge reconstruction loss

                # Load representation learning model
                model = RepresentationLearningModel(input_size, latent_size, view_size, transformer_input_size, hidden_size,
                                                    num_layers,
                                                    num_heads, dropout, lambda0, lambda1, lambda2, lambda3)

                model_path = load_file_from_url(representation_model_path)
                model.load_state_dict(torch.load(model_path))
                test_dct = {}
                for i, t in enumerate(test_data):
                    embedding = rep_model.get_interface_representation(representation_model, t[1], t[2])
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

                gnn_model = GNNModel()
                num_features = len(test[0])
                hidden_size = 512
                num_classes = 2
                learning_rate = 1.0e-3
                num_epochs = 20
                device = "cuda" if torch.cuda.is_available() else "cpu"

                # Load model
                model = GNNModel(num_features, hidden_size, num_classes).to(device)
                model_path = load_file_from_url(validation_model_path)
                model.load_state_dict(torch.load(model_path))

                result_file = open("interface_val_results.txt", "w")

                for i, t in enumerate(test):
                    X, A = t[1], t[2]
                    y_hat = gnn.predict(gnn_model, X, A)
                    result_file.write(f"{t[0]}: {y_hat}")

                result_file.close()
                print("Results are saved as interface_val_results.txt into your current directory.")
            else:
                print(f"Unknown mode: {parser.mode}. Available modes: train, test")
        else:
            print(f"Unknown model: {parser.model}. Available models: RL, GNN")
    else:
        print("You need to provide a path to your dataset directory.")
