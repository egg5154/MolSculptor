from openbabel import openbabel
import numpy as np
from typing import Dict, List, Tuple, Union

def read_pdbqt(file_path: str, is_ligand: bool = False) -> Union[np.ndarray, Dict]:
    """
    read PDBQT(for ligand) / PDB(for protein) file and return coordinates.

    :param file_path: Path to the PDBQT/PDB file.
    :param is_ligand: If True, treat the file as a ligand; otherwise, treat it as a protein.
    :return: 
        - if is_ligand is True, returns an array of ligand atom coordinates (shape: (natm, 3)).
        - if is_ligand is False, returns a dictionary with residue information:
            {
                'chain': str,  # chain identifier
                'residue_number': int,  # residue number
                'residue_name': str,  # residue name
                'C_alpha_coordinates': np.ndarray  # coordinates of Cα atom (shape: (3,))
            }
    """
    
    if is_ligand:
        assert file_path.endswith('.pdbqt')
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("pdbqt")
        
        mol = openbabel.OBMol()
        if not ob_conversion.ReadFile(mol, file_path):
            raise IOError(f"Cannot read file: {file_path}")
        atoms = []
        for atom in openbabel.OBMolAtomIter(mol):
            atoms.append([atom.GetX(), atom.GetY(), atom.GetZ()])
        return np.array(atoms, dtype=np.float32)
    else:
        assert file_path.endswith('.pdb')
        ob_conversion = openbabel.OBConversion()
        ob_conversion.SetInFormat("pdb")
        
        mol = openbabel.OBMol()
        if not ob_conversion.ReadFile(mol, file_path):
            raise IOError(f"Cannot read file: {file_path}")
        protein_residues = {}
        for atom in openbabel.OBMolAtomIter(mol):
            residue = atom.GetResidue()
            if not residue:
                continue
            
            res_name = residue.GetName()
            chain = residue.GetChain()
            res_num = residue.GetNum()
            atom_name = residue.GetAtomID(atom).strip() 
            
            key = f'{chain}-{res_num}-{res_name}'
            res_data = protein_residues.setdefault(key, {'atoms': []})
            """
                res_data: {'atoms': (natm, 3), 'CA': (3,)}
            """
            res_data['atoms'].append([atom.GetX(), atom.GetY(), atom.GetZ()])
            
            if atom_name == 'CA':
                res_data['CA'] = np.array([atom.GetX(), atom.GetY(), atom.GetZ()], dtype=np.float32)

        for res_data in protein_residues.values():
            res_data['atoms'] = np.array(res_data['atoms'], dtype=np.float32)
            
        return protein_residues

def find_pocket_residues(ligand_file: str, protein_file: str, distance_threshold: float = 4.0) -> List[Dict]:
    """
    Get pocket residues based on the distance between ligand and protein residues.

    :param ligand_file: ligand pdbqt file path.
    :param protein_file: protein pdbqt file path.
    :param distance_threshold: Distance threshold to consider a residue as part of the pocket (angstrom).
    :return: list of dict, info of pocket residues.
    """

    lig_atoms = read_pdbqt(ligand_file, is_ligand=True)
    prot_residues = read_pdbqt(protein_file, is_ligand=False)
    
    pocket_residues = []
    for res_key, res_data in prot_residues.items():
        # to ensure the residue has Cα atom
        if 'CA' not in res_data:
            continue
            
        res_atoms_coords = res_data['atoms']  # (n_res_atoms, 3)
        
        # (n_lig_atoms, 1, 3) - (1, n_res_atoms, 3) -> (n_lig_atoms, n_res_atoms, 3)
        dist_matrix = lig_atoms[:, np.newaxis, :] - res_atoms_coords[np.newaxis, :, :]
        
        min_dist = np.min(np.linalg.norm(dist_matrix, axis=-1))
        
        if min_dist <= distance_threshold:
            chain, res_num, res_name = res_key.split('-')
            pocket_residues.append({
                'residue_key': res_key,
                'chain': chain,
                'residue_number': int(res_num),
                'residue_name': res_name,
                'C_alpha_coordinates': res_data['CA']
            })
            
    return pocket_residues

from typing import Dict, List, Tuple
from Bio.PDB import PDBParser, is_aa
import numpy as np

def get_protein_residues(protein_pdb_path: str):

    parser = PDBParser(QUIET=True)

    protein_struct = parser.get_structure("protein", protein_pdb_path)
    serial2chain: Dict[int, str] = {}
    chain_residue_ids: Dict[str, List[int]] = {}

    for model in protein_struct:
        for chain in model:
            cid = chain.id
            seen_res = set()
            chain_ids = []
            for res in chain:

                if is_aa(res, standard=False):
                    resseq = res.get_id()[1]
                    if resseq not in seen_res:
                        seen_res.add(resseq)
                        chain_ids.append(resseq)

                for atom in res:
                    serial2chain[atom.get_serial_number()] = cid
            chain_residue_ids[cid] = np.asarray(chain_ids)
    return chain_residue_ids

import subprocess
def get_fasta(pdb_path):
    try:
        cmd = f'pdb_tofasta -multi {pdb_path}'
        out = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        fasta = out.stdout
        return {l.split('\n')[0][-1]: ''.join(l.split('\n')[1:]) for l in fasta.split('>PDB')[1:]}
    except Exception as e:
        return e

def get_esm_embedding(protein_pdb_path: str, pocket_map: list[dict], chain_res_ids: dict, esm_embedding_dict):
    head_ = os.path.basename(protein_pdb_path) + ':'
    return_esm_embedding = []
    for res in pocket_map:
        chain_ = res['chain']
        loc_ = np.where(chain_res_ids[chain_] == res['residue_number'])
        """
            embedding_dict: {'sequence', 'embedding'}
        """
        emb_ = esm_embedding_dict[(head_ + chain_).strip()]['embedding'][loc_]
        return_esm_embedding.append(emb_)
    return np.concatenate(return_esm_embedding, axis=0).astype(np.float32)

pad_len = 192
def _pad_1d(emb):
    n, d = emb.shape
    assert d == 1280 # esm embedding dim
    mask = np.ones((n,), dtype=np.int32)
    pad_emb = np.pad(emb, ((0, pad_len - n), (0, 0)), mode = 'constant', constant_values=0)
    pad_mask = np.pad(mask, ((0, pad_len - n),), mode = 'constant', constant_values=0)
    return pad_emb, pad_mask
def _pad_2d(mat):
    n, _ = mat.shape
    # print(((0, pad_len - n), (0, pad_len - n)))
    pad_mat = np.pad(mat, ((0, pad_len - n), (0, pad_len - n)), mode = 'constant', constant_values = 0)
    return pad_mat

import os
import pickle as pkl
from scipy.spatial.distance import pdist, squareform
from jax.tree_util import tree_map

esm_script_path = os.path.join(os.path.dirname(__file__), 'esm2_t33_650M_UR50D_fasta.py')
def main(args):

    esm_env_python_path = args.esm_env_python_path
    esm_model_path = args.esm_model_path
    with open(args.name_list_path, 'r') as f:
        name_list = f.readlines()
        name_list = [n.strip().split() for n in name_list]
    
    chain_res_ids_list = []
    pocket_map_list = []
    write_lines = []
    for protein_pdb_path, ligand_pdbqt_path in name_list:
        _chain_res_ids = get_protein_residues(protein_pdb_path)
        _pocket_residues_map = find_pocket_residues(ligand_pdbqt_path, protein_pdb_path, distance_threshold = 8.0)
        chain_res_ids_list.append(_chain_res_ids)
        pocket_map_list.append(_pocket_residues_map)
        _pocket_chains = {_res['chain'] for _res in _pocket_residues_map}
        _fasta_dict = get_fasta(protein_pdb_path)
        head_ = os.path.basename(protein_pdb_path) + ':'
        for k in _pocket_chains:
            write_lines.append('>' + head_ + k + '\n' + _fasta_dict[k] + '\n')

    fasta_path = os.path.join(args.save_path, 'inputs.fasta')
    esm_output_path = os.path.join(args.save_path, 'esm_output.pkl')
    esm_log_path = os.path.join(args.save_path, 'esm_log.txt')
    with open(fasta_path, 'w') as f:
        for line in write_lines: f.write(line)
    cmd = f'{esm_env_python_path} {esm_script_path} --fasta_path {fasta_path} --output_pkl_path {esm_output_path} --model_path {esm_model_path}'
    with open(esm_log_path, 'w') as f:
        subprocess.run(cmd, shell=True, check=True, stderr=f, stdout=f)
    
    with open(esm_output_path, 'rb') as f:
        esm_output = pkl.load(f)
    esm_embedding = []
    distance_matrix = []
    residue_mask = []
    for i in range(len(name_list)):
        esm_emb = get_esm_embedding(name_list[i][0], pocket_map_list[i], chain_res_ids_list[i], esm_output)
        coords_array = np.asarray([d['C_alpha_coordinates'] for d in pocket_map_list[i]])
        dist_mat = squareform(pdist(coords_array, 'euclidean')).astype(np.float32)
        esm_emb, mask = _pad_1d(esm_emb)
        dist_mat = _pad_2d(dist_mat)
        esm_embedding.append(esm_emb)
        distance_matrix.append(dist_mat)
        residue_mask.append(mask)
    
    esm_embedding = np.stack(esm_embedding, axis=0)
    distance_matrix = np.stack(distance_matrix, axis=0)
    residue_mask = np.stack(residue_mask, axis=0)

    with open(os.path.join(args.save_path, 'pocket_features.pkl'), 'wb') as f:
        _data = {'esm_embedding': esm_embedding, 'residue_mask': residue_mask, 'distance_matrix': distance_matrix,}
        pkl.dump(_data, f)
    print('Pocket features saved to:', args.save_path)
    print(tree_map(np.shape, _data))
    print("Index:")
    for i, line in enumerate(name_list):
        protein_pdb_path, ligand_pdbqt_path = line
        print(i, protein_pdb_path)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get pocket features from PDBQT files.")
    parser.add_argument('--name_list_path', type=str, required=True, help='Path to the file containing protein and ligand PDBQT paths.')
    parser.add_argument('--save_path', type=str, required=True, help='Directory to save the output files.')
    parser.add_argument('--esm_env_python_path', type=str, help='Path to the Python executable in the ESM environment.')
    parser.add_argument('--esm_model_path', type=str, help='Path to the pretrained ESM model.')
    args = parser.parse_args()
    
    main(args)

