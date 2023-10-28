import os
import re
from dictionaries import MAX_ASA


class ASACalculator:

    def __init__(self, pdb_path, interfaceID, chain_1, chain_2):
        self.pdb_path = pdb_path
        self.interfaceID = interfaceID
        self.chain_1 = chain_1
        self.chain_2 = chain_2
        self.complex_asa = dict()
        self.relative_complex_asa = dict()

    # Calculate RelASA
    def parse_naccess_file(self):
        pdb_id = self.interfaceID[0:4]
        file = pdb_id + self.interfaceID[4]
        nfile = open(self.pdb_path + "/" + self.interfaceID.lower() + ".naccess", "r")
        line = nfile.readline()
        while line:
            if line.startswith("RES"):
                resPosition = line[9:14].replace(" ", "")
                chain = line[8]
                if chain == self.chain_1.id:
                    residue = self.chain_1[int(re.sub("\D", "", resPosition))]
                else:
                    residue = self.chain_2[int(re.sub("\D", "", resPosition))]
                temp = line[14:].strip().split()
                ASA = float(temp[0])
                self.complex_asa[residue] = ASA
                self.relative_complex_asa[residue] = ASA / MAX_ASA[residue.get_resname()] * 100
            line = nfile.readline()
        nfile.close()
        return

    def calc_sasa(self, structure):
        pdb_file = structure.lower()
        pdb_ID = pdb_file
        pdb_file = "%s.pdb" % pdb_file
        cmd = "%s/naccess %s > out.txt 2>error.txt" % ("naccess", pdb_file)
        os.system("cp %s/%s.pdb naccess/%s.pdb " % (self.pdb_path, pdb_ID, pdb_ID))
        os.chmod("naccess/%s.pdb" % (pdb_ID), 777)
        os.chdir("naccess")
        os.system(cmd)
        os.chdir("")
        os.system("cp naccess/%s.rsa %s/%s.naccess" % (pdb_ID, self.pdb_path, pdb_ID))
        os.system("chmod -R 777 %s" % self.pdb_path)
        os.system("chmod -R 777 naccess")
        os.system("rm naccess/*.log naccess/*.asa naccess/*.pdb naccess/*.rsa")
        return

    def run_naccess(self):
        pdb_id = self.interfaceID[0:4]
        for structure in [pdb_id + self.interfaceID[4], pdb_id + self.interfaceID[5], self.interfaceID]:
            self.calc_sasa(structure)
        self.parse_naccess_file()
