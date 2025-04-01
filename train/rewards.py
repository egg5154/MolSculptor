"""
    In this script, we store the reward functions for guided diffusion.
    NOTE: Higher score is better!!!
"""

import jax
import numpy as np
import jax.numpy as jnp
import os
import sys
import subprocess
import datetime

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Crippen, QED, AllChem, DataStructs
from rdkit.Contrib.SA_Score import sascorer # type: ignore
# from .sascorer import calculateScore

def LogP_reward(molecule_dict):
    smiles = molecule_dict['smiles']
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [Crippen.MolLogP(m) for m in mols]

def QED_reward(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [QED.qed(m) for m in mols]

def SA_reward(smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    return [sascorer.calculateScore(m) for m in mols]

def tanimoto_sim(smiles1, smiles2):
    scores = []
    assert smiles1.shape[0] == smiles2.shape[0]
    for i in range(smiles1.shape[0]):
        mol1 = Chem.MolFromSmiles(smiles1[i], sanitize = True)
        mol2 = Chem.MolFromSmiles(smiles2[i], sanitize = True)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits = 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits = 2048)
        score = DataStructs.FingerprintSimilarity(fp1, fp2)
        scores.append(score)
    return scores

def smi2pdbqt(smi, pdbqt_path):

    command = f"obabel -:\"{smi}\" -opdbqt -O {pdbqt_path} --gen3d -h"
    try:
        subprocess.run(command, shell = True, check = True,
            stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("Error during obabel execution:")
        print(e.stderr)
        raise e

def dsdp_reward(smi: str, cached_file_path: str, dsdp_script_path: str):
    smi_save_dir = os.path.join(cached_file_path, f'temp-ligand.pdbqt')
    smi2pdbqt(smi, smi_save_dir)
    out_dir = os.path.join(cached_file_path, f'temp-dock.pdbqt')
    log_dir = os.path.join(cached_file_path, f'temp-log.log')
    cmd = ['bash', dsdp_script_path, smi_save_dir, out_dir, log_dir]
    subprocess.run(cmd, check=True, shell=True)
    with open(log_dir, 'r') as f:
        lines = f.readlines()
        ss = [float(s.split()[-1]) for s in lines]
    return ss[0]

def dsdp_batch_reward(
        smiles: np.ndarray, 
        cached_file_path: str, dsdp_script_path: str,
        gen_lig_pdbqt: bool = True):
    ### smiles: (N,)
    scores = []
    if gen_lig_pdbqt:
        name_list = []
        print("Generating pdbqt files...")
        for i in tqdm(range(smiles.shape[0])):
            smi = smiles[i]
            smi_save_dir = os.path.join(
                cached_file_path, f'ligands/{i}.pdbqt')
            smi2pdbqt(smi, smi_save_dir)
            name_list.append(f'{i}.pdbqt')
        ## create name list file
        name_list_path = os.path.join(cached_file_path, 'name_list.txt')
        with open(name_list_path, 'w') as f:
            f.write('\n'.join(name_list))
    else:
        name_list_path = os.path.join(cached_file_path, 'name_list.txt')
        with open(name_list_path, 'r') as f:
            name_list = f.readlines()
        name_list = [s.strip() for s in name_list]
    ## run dsdp script
    print("Estimating DSDP reward...")
    t_0 = datetime.datetime.now()

    ## seq run
    for i in tqdm(range(len(name_list))):
        out_dir = os.path.join(cached_file_path, f'outputs/{i}.out')
        log_dir = os.path.join(cached_file_path, f'logs/{i}.log')
        lig_dir = os.path.join(cached_file_path, f'ligands/{i}.pdbqt')
        cmd = f'{dsdp_script_path} {lig_dir} {out_dir} {log_dir}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t_1 = datetime.datetime.now()
    print(f"Time used: {t_1 - t_0}")
    print("Reading log files...")
    for i in tqdm(range(len(name_list))):
        log_dir = os.path.join(cached_file_path, f'logs/{i}.log')
        with open(log_dir, 'r') as f:
            lines = f.readlines()
            ss = [float(s.split()[-1]) for s in lines]
        scores.append(ss[0]) ## the highest
    return scores