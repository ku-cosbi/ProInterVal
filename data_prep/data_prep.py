import numpy as np
import os
from Bio.PDB import *
from dictionaries import *
import torch
from scipy.spatial.distance import cdist
from asa_calculator import *
from pp_calculator import *


def parse(structure, pdb, chain_1, chain_2):
    os.system("chmod -R 777 PDBs")

    class Chain1Select(Select):
        def accept_chain(self, chain):
            if chain.id == chain_1:
                return 1
            else:
                return 0

    class Chain2Select(Select):
        def accept_chain(self, chain):
            if chain.id == chain_2:
                return 1
            else:
                return 0

    class Chain12Select(Select):
        def accept_chain(self, chain):
            if (chain.id == chain_1) or (chain.id == chain_2):
                return 1
            else:
                return 0

    io = PDBIO()
    io.set_structure(structure)
    io.save("data/%s%s.pdb" % (pdb.lower(), chain_1.lower()), Chain1Select())
    io.save("data/%s%s.pdb" % (pdb.lower(), chain_2.lower()), Chain2Select())
    io.save("data/%s%s.pdb" % (pdb.lower(), chain_1.lower() + chain_2.lower()), Chain12Select())
    return


def parse_pdb(pdb_id, chain_id_1, chain_id_2, file_path):
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure('myPDBStructure', "PDBs/%s.pdb" % (pdb_id))
    parse(structure, pdb_id, chain_id_1, chain_id_2)
    structure = pdb_parser.get_structure('myPDBStructure',
                                         "PDBs/%s%s.pdb" % (pdb_id.lower(), chain_id_1.lower() + chain_id_2.lower()))
    model = structure[0]
    chain_1 = model[chain_id_1]
    chain_2 = model[chain_id_2]
    residue_list = Selection.unfold_entities(structure, 'R')
    return residue_list, chain_1, chain_2


def get_neighbors(residue_list, interface_residues, cutoff_distance):
    # Get neighbors of interface residues within 6 Å
    ns = NeighborSearch(residue_list)
    nearby_residues = set()
    for res in interface_residues:
        nearby_residues.update(ns.search(residue_list[res]['CA'].get_coord(), cutoff_distance, level='R'))
    return nearby_residues


def get_interface_residues(residue_list, chain1, chain2, cutoff_distance):
    # Extracts the residues in the interface
    interface_residues = []
    for residue1 in chain1:
        for residue2 in chain2:
            if residue1.get_id()[0] == " " and residue2.get_id()[0] == " ":
                if residue1["CA"] - residue2["CA"] <= cutoff_distance:
                    interface_residues.append(residue1)
                    interface_residues.append(residue2)
    nearby_residues = get_neighbors(residue_list, interface_residues, 6.0)
    interface_residues = list(nearby_residues) + interface_residues
    return interface_residues


def get_residue_features(path, pdb_id, chain_1, chain_2, residue_list):
    node_features = {}
    interface_id = pdb_id + "_" + chain_1.id + "_" + chain_2.id
    asa_calculator = ASACalculator(path, interface_id, chain_1, chain_2)
    asa_calculator.run_naccess()
    pp_calculator = PPCalculator(residue_list)
    pp_calculator.calculate_pp()
    for res in residue_list:
        res_type = THREEtoONE[res.get_resname()]
        polarity = POLARITY[res.get_resname()]
        charge = CHARGE[res.get_resname()]
        relative_asa = asa_calculator.relative_complex_asa[res]
        pp = pp_calculator.pair_potentials[res]
        phi = res.get_phi()
        psi = res.get_psi()
        node_features[res.id[1]] = np.array([res_type, polarity, charge, relative_asa, phi, psi])
    return node_features


def generate_adjacency_matrix(interface_residues, sequence_cutoff, distance_cutoff, k):
    num_residues = len(interface_residues)
    adjacency_matrix = np.zeros((num_residues, num_residues))

    coords = np.array([residue["CA"].get_coord() for residue in interface_residues])
    # Calculate the pairwise distance between all residues
    distances = cdist(coords, coords)

    for i, residue1 in enumerate(interface_residues):
        knn_indices = np.argsort(distances[i])[:k]

        for j, residue2 in enumerate(interface_residues):
            edge = np.zeros(3)
            if i == j:
                continue
            # sequence edge
            if abs(residue1.get_id()[1] - residue2.get_id()[1]) <= sequence_cutoff:
                edge[0] = 1

            # radius edge
            distance = np.linalg.norm(
                interface_residues[i]['CA'].get_coord() - interface_residues[j]['CA'].get_coord())
            if distance <= distance_cutoff:
                edge[1] = 1

            # KNN edge
            if j in knn_indices:
                edge[2] = 1
            adjacency_matrix[i, j] = edge
            adjacency_matrix[j, i] = edge
    return adjacency_matrix


def generate_graph(path, pdb_id, residue_list, chain_1, chain_2):
    # Extract interface and nearby residues
    interface_residues = get_interface_residues(residue_list, chain_1, chain_2, 0.5)

    # Calculate node features
    residue_features = get_residue_features(path, pdb_id, chain_1, chain_2, interface_residues)

    # Get the node feature values as a list of lists
    node_features = [v for k, v in residue_features.items()]

    # Convert the list of lists to a tensor
    node_feature_tensor = torch.tensor(node_features, dtype=torch.float)
    adj_matrix_tensor = torch.from_numpy(generate_adjacency_matrix(interface_residues, 3, 10.0, 10))
    return node_feature_tensor, adj_matrix_tensor


def prepare_data():
    data = []
    directory = '../../../../../datasets/pdb/interfaces/pdb'
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            for path, _, files in os.walk(os.path.join(root, d)):
                for filename in files:
                    interface_id = filename.split(".pdb")[0]
                    pdb_id, chain_id_1, chain_id_2 = interface_id.split("_")
                    residue_list, chain_1, chain_2 = parse_pdb(pdb_id, chain_id_1, chain_id_2, path)
                    X, A = generate_graph(path, pdb_id, residue_list, chain_1, chain_2)
                    data.append((X, A))
    return data


def get_deepinterface_data():
    data = []
    directory = '../../cosbi/backup/kuacc/ahakouz17/workfolder/deepinter/deepinter_data/'
    for root, dirs, _ in os.walk(directory):
        for d in dirs:
            for path, _, files in os.walk(os.path.join(root, d)):
                for filename in files:
                    interface_id = filename.split(".pdb")[0]
                    pdb_id, chain_id_1, chain_id_2 = interface_id.split("_")
                    residue_list, chain_1, chain_2 = parse_pdb(pdb_id, chain_id_1, chain_id_2, path)
                    label = 1 if d == 'pos' else 0
                    X, A = generate_graph(path, pdb_id, residue_list, chain_1, chain_2)
                    data.append((interface_id, X, A, label))
    return data

