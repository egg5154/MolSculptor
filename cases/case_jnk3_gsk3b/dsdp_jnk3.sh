#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
"${SCRIPT_DIR}/../../dsdp/DSDP"\
	--ligand $1\
	--protein $( dirname "${BASH_SOURCE[0]}" )/jnk3-3oy1.pdbqt\
	--box_min -36.000 -6.000 -38.000 \
	--box_max -6.000 24.000 -18.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3