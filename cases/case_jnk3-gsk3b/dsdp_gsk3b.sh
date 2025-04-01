#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
"${SCRIPT_DIR}/../../dsdp/DSDP"\
	--ligand $1\
	--protein $(dirname "${BASH_SOURCE[0]}")/gsk3b-6y9s.pdbqt\
	--box_min -15.000 6.000 10.000 \
	--box_max 15.000 36.000 40.000 \
	--exhaustiveness 384 --search_depth 40 --top_n 4\
	--out $2\
	--log $3