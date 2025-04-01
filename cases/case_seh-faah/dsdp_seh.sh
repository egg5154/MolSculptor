#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
"${SCRIPT_DIR}/../../dsdp/DSDP"\
	--ligand $1\
	--protein $( dirname "${BASH_SOURCE[0]}" )/seh-3wke.pdbqt\
	--box_min -30.000 -21.000 50.000 \
	--box_max -5.000 9.000 86.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3