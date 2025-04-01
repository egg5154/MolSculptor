#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
"${SCRIPT_DIR}/../../dsdp/DSDP"\
	--ligand $1\
	--protein $( dirname "${BASH_SOURCE[0]}" )/faah-2wj1.pdbqt\
	--box_min 5.000 -31.000 16.000 \
	--box_max 36.000 -1.000 46.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3