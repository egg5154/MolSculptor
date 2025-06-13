# MolSculptor: a training-free framework for multi-site inhibitor design
This is the github repo for the paper *MolSculptor: a diffusion-evolution framework for multi-site inhibitor design*, which is preprinted at [ChemRxiv](https://doi.org/10.26434/chemrxiv-2025-v4758).

## Installation
Running example scripts in [cases](./cases) requires:
* python==3.12
* jax==0.4.28, jaxlib==0.4.28
* flax==0.8.3
* ml-collections==0.1.1
* rdkit==2023.9.6
* openbabel==3.1.1

We also provide [requirements.txt](./requirements.txt) to make sure you can quickly create a compatible environment by the following steps:
```
conda create -n molsculptor_env python=3.12
conda activate molsculptor_env
pip install -r requirements.txt
conda install openbabel=3.1.1 -c conda-forge
```
Our configuration includes Ubuntu 22.04 (GNU/Linux x86_64), NVIDIA A100-SXM4-80GB, CUDA 12.2 and Anaconda 23.7.2.

After setting up the Python environment, we also need to install [DSDP](https://github.com/PKUGaoGroup/DSDP), a GPU-accelerated tool for molecular docking:
```
cd dsdp
git clone https://github.com/PKUGaoGroup/DSDP.git DSDP_main
cd DSDP_main/DSDP_redocking/
make
cp DSDP ../../
cd ../../../
```
Finally we need to get the model parameters for auto-encoder model and diffusion transformer model:
```
wget -P checkpoints/auto-encoder https://zenodo.org/records/15123602/files/ae_params_denovo.pkl
wget -P checkpoints/auto-encoder https://zenodo.org/records/15123602/files/ae_params_opt.pkl
wget -P checkpoints/diffusion-transformer https://zenodo.org/records/15123602/files/dit_params_denovo.pkl
wget -P checkpoints/diffusion-transformer https://zenodo.org/records/15123602/files/dit_params_opt.pkl
```
The install time depends on your Internet speed, approximately ranging from 0.5 to 1 hour.
## Molsculptor's current capabilities
The test cases in our paper are saved in [cases](./cases), including three dual-target inhibitor design tasks and one PI3K selective inhibitor design task.
### Dual-target inhibitor design
We tested the molecular optimization capability for MolSculptor in three dual-target inhibitor design tasks:
* c-Jun N-terminal kinase 3 and Glycogen synthase kinase-3 beta (JNK3/GSK3beta)
```
bash cases/case_jnk3-gsk3b/opt_jnk3-gsk3b.sh
```
* Androgen receptor and glucocorticoid receptor (AR/GR)
```
bash cases/case_ar-gr/opt_ar-gr.sh
```
* Soluble epoxide hydrolase and fatty acid amide hydrolase (sEH/FAAH)
```
bash cases/case_seh-faah/opt_seh-faah.sh
```
### Selective inhibitor *de novo* design
```
bash cases/case_pi3k/denovo_pi3k.sh
```
The runtime is approximately 8 hours for optimization cases and 16 hours for PI3K de novo design case. 
### How to build your own case
#### Dual inhibitor optimization
For dual-inhibitor optimization task, you will need:
* `.pdbqt`files for target proteins
* DSDP docking scripts
* initial molecules (its SMILES, molecular graphs and docking scores)
##### Creating `.pdbqt` file for target proteins
You can use [openbabel](https://github.com/openbabel/openbabel) to create the protein `.pdbqt` file from a sanitized `.pdb` file:
```
obabel -ipdb xxx.pdb -opdbqt xxx.pdbqt -p 7.4
```
##### Creating DSDP docking scripts
The general script format is as follows (assume this script is in `cases/your_own_cases` folder):
```
#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
"${SCRIPT_DIR}/../../dsdp/DSDP"\
	--ligand $1\
	--protein $SCRIPT_DIR/xxx.pdbqt\
	--box_min [x_min] [y_min] [z_min] \
	--box_max [x_max] [y_max] [z_max] \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3
```
Where the `--protein` argument is for the target `.pdbqt` file, the `--box_min` and `--box_max` argument define the sampling cubic region.

##### Creating initial molecule input file
You can use [make-init-molecule.ipynb](./tests/make-init-molecule.ipynb) to create `init_search_molecule.pkl`. The `.pkl` file will be saved in `tests/init-molecule`.

##### Choosing a suitable noise schedule
You can use [noising-denoising_test.py](./tests/noising-denoising_test.py) and [noising-denoising_analysis.ipynb](./tests/noising-denoising_analysis.ipynb) to exmaine the relationship between diffusion timestep and molecular similarity, validity and other optimization/generation related metrics.

##### Create the main script
The main script contains the following required arguments:
* `--params_path` and  `--config_path`: the diffusion transformer parameter & config path
* `--logger_path`: the logging file path
* `--save_path`: path for saving the optimized molecules
* `--dsdp_script_path_1` & `--dsdp_script_path_2`: paths for target protein docking scripts
* `--random_seed` & `--np_random_seed`: random seed for `numpy` and `jax.numpy`
* `--total_step`: total evolution steps
* `--device_batch_size`: population size for evolution algotithm
* `--n_replicate`: number of offsprings for one parent molecule
* `--t_min` & `--t_max`: min/max diffusion timestep
* `--vae_config_path` & `--vae_params_path`: the auto-encoder parameter & config path
* `--alphabet_path`: the path for SMILES alphabet, default set is [smiles_alphabet.pkl](./train/smiles_alphabet.pkl)
* `--init_molecule_path`: the path for initial molecule input file
* `--sub_smiles`: the SMILES string for the substructure you want to retain during optimization

## Training
The training scripts are located in [training_scripts](./training_scripts) and the example training data (pre-processed) is in [zenodo](https://zenodo.org/records/15653724/files/training_data.tar.gz?download=1).

Before you begin, download and unpack the pre-processed example dataset into the root of the MolSculptor repository:

```bash
wget https://zenodo.org/records/15653724/files/training_data.tar.gz?download=1
tar -xvf training_data.tar.gz
```
To launch the AE pre-training script using MPI-style arguments:
```
bash training_scripts/pretrain_ae.sh [YOUR IP ADDRESS OF PROCESS 0] [NUM PROCESSES] [RANK]
## for example, if the model is trained on a single host, with ip 128.5.1.1
bash training_scripts/pretrain_ae.sh 128.5.1.1 1 0
## if the model is trained on multi-hosts (2 hosts for example)
# on host A
bash training_scripts/pretrain_ae.sh 128.5.1.1 2 0
# on host B
bash training_scripts/pretrain_ae.sh 128.5.1.1 2 1
```
To train the diffusion-transformer model:
```
bash training_scripts/train_dit.sh [YOUR IP ADDRESS OF PROCESS 0] [NUM PROCESSES] [RANK]
```

## Citation
```
@article{li2025molsculptor,
  title={MolSculptor: a diffusion-evolution framework for multi-site inhibitor design},
  author={Li, Yanheng and Lin, Xiaohan and Hao, Yize and Zhang, Jun and Gao, Yi Qin},
  year={2025}
}
```
## Contact
For questions or further information, please contact [gaoyq@pku.edu.cn](gaoyq@pku.edu.cn) or [jzhang@cpl.ac.cn](jzhang@cpl.ac.cn).
