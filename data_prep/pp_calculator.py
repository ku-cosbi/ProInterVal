from dictionaries import CONTACT_POTENTIALS, PP_indices, DICT_ATOM
import math


def distance_calculation(vector_1, vector_2):
    x1 = vector_1[0]
    y1 = vector_1[1]
    z1 = vector_1[2]

    x2 = vector_2[0]
    y2 = vector_2[1]
    z2 = vector_2[2]

    distance = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2) + ((z2 - z1) ** 2))
    return distance


def get_com(residue):
    x_total = 0
    y_total = 0
    z_total = 0
    if len(residue.get_resname()) > 3:
        resname = residue.get_resname()[1:4]
    else:
        resname = residue.get_resname()
    num_atoms = len(DICT_ATOM[resname])
    for atom_id in DICT_ATOM[resname]:
        atom = residue[atom_id]
        x_total += atom.get_coord()[0]
        y_total += atom.get_coord()[1]
        z_total += atom.get_coord()[2]
    return [x_total / num_atoms, y_total / num_atoms, z_total / num_atoms]


class PPCalculator:

    def __init__(self, residue_list):
        self.residue_list = residue_list
        self.pair_potentials = dict()

    def extract_neighbors(self, residue):
        com_residue = get_com(residue)
        neighbors = []
        for neighbor in self.residue_list:
            try:
                com_neighbor = get_com(neighbor)
                if residue != neighbor:
                    distance = distance_calculation(com_residue, com_neighbor)
                    if residue.get_parent().id == neighbor.get_parent().id:  # they are in the same chain
                        if abs(residue.get_id()[1] - neighbor.get_id()[1]) >= 4:
                            if distance <= 7.0:
                                neighbors.append(neighbor)
                    else:
                        if distance <= 7.0:
                            neighbors.append(neighbor)
            except:
                pass
        return neighbors

    def contact_potentials(self, residue):
        neighbors = self.extract_neighbors(residue)
        sum_pp = 0.0
        for neighbor in neighbors:
            indexI = PP_indices[residue.get_resname()]
            indexJ = PP_indices[neighbor.get_resname()]
            if CONTACT_POTENTIALS[indexI][indexJ] != 0:
                sum_pp += CONTACT_POTENTIALS[indexI][indexJ]
            else:
                sum_pp += CONTACT_POTENTIALS[indexJ][indexI]
        return sum_pp

    def calculate_pp(self):
        for residue in self.residue_list:
            try:
                pp = self.contact_potentials(residue)
                self.pair_potentials[residue] = abs(pp)
            except:
                pass
