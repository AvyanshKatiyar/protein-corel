#!/bin/bash 
#Loop 1
#NUM CPUS = PREFIX_CPU
VAR=1000
PREFIX_CPU=3_CHAINS_"$VAR"_E_10_B_30_Multiplier_1_tanh #remember too change ncpus

# Create a folder with the same name as PREFIX_CPU
mkdir -p "$PREFIX_CPU"


cp train7.sh "$PREFIX_CPU/train.sh"

cp train_corel.py "$PREFIX_CPU/train.py"

sed -i 's/NUM_CHAINS/'$VAR'/g' "$PREFIX_CPU/train.py"
sed -i 's/NUM_CPUS/'$PREFIX_CPU'/g' "$PREFIX_CPU/train.py"

sed -i 's/DIRECTORY/'$PREFIX_CPU'/g' "$PREFIX_CPU/train.sh"

cd "$PREFIX_CPU/"
qsub -N test$VAR -q low -lselect=1:ncpus=8 -P chemical -l walltime=96:00:00 train.sh
