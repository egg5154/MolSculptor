#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
"${SCRIPT_DIR}/../../dsdp/DSDP"\
	--ligand $1\
	--protein $( dirname "${BASH_SOURCE[0]}" )/bike-4w9w.pdbqt\
	--box_min -30.000 -40.000 -20.000 \
	--box_max 0.000 -10.000 10.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3