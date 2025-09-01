"""
    In this script, we store the reward functions for guided diffusion.
    NOTE: Higher score is better!!!
"""

import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import os
import time
import sys
import subprocess
import datetime
import multiprocessing

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

def smi2pdbqt_obabel(smi, pdbqt_path):

    command = f"obabel -:\"{smi}\" -opdbqt -O {pdbqt_path} --gen3d -h"
    try:
        subprocess.run(command, shell = True, check = True,
            stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("Error during obabel execution:")
        print(e.stderr)
        raise e

import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path
from rdkit.Chem import AllChem
from typing import Optional

def _embed_worker(molblock: str, seed: int, max_attempts: int, result_q: mp.Queue):
    """
    MolBlock -> RDKit mol -> Embed -> Return(status code, MolBlock)
    """
    try:
        mol = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=True)
        if mol is None:
            result_q.put(("error", "MolFromMolBlock failed"))
            return

        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.maxAttempts = max_attempts

        code = AllChem.EmbedMolecule(mol, params)
        # code == 0 -> success; code == -1 -> failure
        result_q.put(("ok", (code, Chem.MolToMolBlock(mol))))
    except Exception as e:
        result_q.put(("error", repr(e)))

def embed_with_hard_timeout(
    mol: Chem.Mol,
    *,
    seed: int = 42,
    max_attempts_first: int = 20,
    max_attempts_retry: int = 200,
    timeout_sec_first: float = 10.0,
    timeout_sec_retry: float = 15.0
) -> Chem.Mol:

    base_block = Chem.MolToMolBlock(mol)

    def _run_once(max_attempts: int, timeout_sec: float) -> Optional[Chem.Mol]:
        q: mp.Queue = mp.Queue(1)
        p = mp.Process(target=_embed_worker, args=(base_block, seed, max_attempts, q))
        p.start()
        p.join(timeout_sec)

        if p.is_alive():
            # timeout
            p.terminate()
            p.join()
            return None

        # get results
        try:
            status, payload = q.get_nowait()
        except Exception:
            raise RuntimeError("ETKDG failed without any output")

        if status == "error":
            raise RuntimeError(f"ETKDG error: {payload}")

        code, out_block = payload
        if code != 0:
            return False  # emb failed but not timeout
        out_mol = Chem.MolFromMolBlock(out_block, removeHs=False, sanitize=True)
        if out_mol is None:
            raise RuntimeError("MolBlock parsing failed")
        return out_mol

    # first attempt
    res = _run_once(max_attempts_first, timeout_sec_first)
    if res is None:
        raise RuntimeError(f"ETKDG timeout (>{timeout_sec_first} s)")
    if res is not False:
        return res

    # second attempt with more tries
    res = _run_once(max_attempts_retry, timeout_sec_retry)
    if res is None:
        raise RuntimeError(f"ETKDG timeout (>{timeout_sec_retry} s)")
    if res is False:
        raise RuntimeError("ETKDG failed")
    return res

def smi2pdbqt_meeko(
    smi: str,
    pdbqt_path: str,
    *,
    meeko_cli: str = "mk_prepare_ligand.py",
    timeout: int = 600,
    config_file: str | None = None,
    rigid_macrocycles: bool = False,
    keep_chorded_rings: bool = False,
    keep_equivalent_rings: bool = False,
    min_ring_size: int | None = None,
    remove_smiles: bool = False,
):

    exe = shutil.which(meeko_cli)
    if exe is None:
        raise RuntimeError(
            f"Cannot find {meeko_cli}, please install meeko & rdkit:\n"
            "    pip install meeko rdkit\n"
        )

    out_path = Path(pdbqt_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RDKit: SMILES -> Mol -> addH -> 3D emb -> optimization -> write SDF
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Cannot parse SMILES string.")
    mol = Chem.AddHs(mol)
    mol = embed_with_hard_timeout(mol, seed=42, timeout_sec_first=5., timeout_sec_retry=10.)

    # Optimization
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)

    with tempfile.TemporaryDirectory() as td:
        sdf_file = Path(td) / "ligand.sdf"
        w = Chem.SDWriter(str(sdf_file))
        mol.SetProp("_Name", "ligand")
        w.write(mol)
        w.close()

        # Meeko CLI: SDF -> PDBQT
        cmd = [exe, "-i", str(sdf_file), "-o", str(out_path)]
        if config_file:
            cmd += ["-c", str(config_file)]
        if rigid_macrocycles:
            cmd.append("--rigid_macrocycles")
        if keep_chorded_rings:
            cmd.append("--keep_chorded_rings")
        if keep_equivalent_rings:
            cmd.append("--keep_equivalent_rings")
        if min_ring_size is not None:
            cmd += ["--min_ring_size", str(min_ring_size)]
        if remove_smiles:
            cmd.append("--remove_smiles")

        try:
            res = subprocess.run(
                cmd, check=True, timeout=timeout,
                capture_output=True, text=True
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"mk_prepare_ligand.py timeout (>{timeout} s) \nCMD: {' '.join(cmd)}") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "mk_prepare_ligand.py failed: \n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{e.stdout}\n"
                f"STDERR:\n{e.stderr}"
            ) from e

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("No valid output .pdbqt file generated.")

def dsdp_reward(smi: str, cached_file_path: str, dsdp_script_path: str, use_obabel: bool = True):
    smi2pdbqt = smi2pdbqt_obabel if use_obabel else smi2pdbqt_meeko
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
        gen_lig_pdbqt: bool = True,
        use_obabel: bool = True,):
    ### smiles: (N,)
    smi2pdbqt = smi2pdbqt_obabel if use_obabel else smi2pdbqt_meeko
    scores = []
    if gen_lig_pdbqt:
        name_list = []
        err_list = []
        print("Generating pdbqt files...")
        for i in tqdm(range(smiles.shape[0])):
            smi = smiles[i]
            smi_save_dir = os.path.join(
                cached_file_path, f'ligands/{i}.pdbqt')
            try:
                smi2pdbqt(smi, smi_save_dir)
                name_list.append(f'{i}.pdbqt')
            except:
                err_list.append(f'{i}.pdbqt')
        print(f"Failed to generate {len(err_list)} pdbqt files.")
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
    for pdbqt_i in tqdm(name_list):
        idx = pdbqt_i.split('.')[0]
        out_dir = os.path.join(cached_file_path, f'outputs/{idx}.out')
        log_dir = os.path.join(cached_file_path, f'logs/{idx}.log')
        lig_dir = os.path.join(cached_file_path, f'ligands/{idx}.pdbqt')
        cmd = f'{dsdp_script_path} {lig_dir} {out_dir} {log_dir}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    t_1 = datetime.datetime.now()
    print(f"Time used: {t_1 - t_0}")
    print("Reading log files...")
    for i in tqdm(range(smiles.shape[0])):
        try:
            log_dir = os.path.join(cached_file_path, f'logs/{i}.log')
            with open(log_dir, 'r') as f:
                lines = f.readlines()
                ss = [float(s.split()[-1]) for s in lines]
            scores.append(ss[0]) ## the highest
        except:
            scores.append(0.0) # zero padding for failed ones
    return scores