# MolSculptor: a training-free framework for multi-site inhibitor design
This is the github repo for the paper *Multi-site inhibitor design using a diffusion-evolution framework*, which is preprinted at [arXiv]().

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
pip install -r requirements.txt
conda install openbabel=3.1.1 -c conda-forge
```
Ourconfiguration includes Ubuntu 22.04 (GNU/Linux x86_64), NVIDIA A100-SXM4-80GB, CUDA 12.2 and Anaconda 23.7.2.

After setting up the Python environment, we also need to install [DSDP](https://github.com/PKUGaoGroup/DSDP), a GPU-accelerated tool for molecular docking:
```
cd dsdp
git clone https://github.com/PKUGaoGroup/DSDP.git DSDP_main
cd DSDP_main/DSDP_redocking/
make
cp DSDP ../../
```
Finally we need to get the model parameters for auto-encoder model and diffusion transformer model:
```

```
## Molsculptor's current capabilities
The test cases in out paper is saved in [cases](./cases), including three dual-target inhibitor design tasks and one PI3K selective inhibitor design task.
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
### How to build your own case
#### Dual inhibitor optimization
For dual-inhibitor optimization task, you will need:
* `.pdbqt`files for target proteins
* DSDP docking scripts
* initial molecules (its SMILES, molecular graphs and docking scores)
##### Creating `.pdbqt` file for target proteins
You can use [openbabel](https://github.com/openbabel/openbabel) to create the protein `.pdbqt` file from a sanitized `.pdb` file:
```
obabel -ipdb xxx.pdb -opdbqt xxx.pdbqt -h
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
You can use [noising-denoising_test.py](./tests/noising-denoising_test.py) and [noising-denoising_analysis.py](./tests/noising-denoising_analysis.ipynb) to exmaine the relationship between diffusion timestep and molecular similarity, validity and other optimization/generation related metrics.

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

## Citation

## Contact
For questions or further information, please contact [gaoyq@pku.edu.cn](gaoyq@pku.edu.cn) or [jzhang@cpl.ac.cn](jzhang@cpl.ac.cn).
