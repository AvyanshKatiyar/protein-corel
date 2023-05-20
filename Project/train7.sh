#!/bin/bash

cd /home/chemical/btech/ch1190960/scratch/BTP/pdnet-master/DIRECTORY

# Load the Anaconda Python 3 module
module load apps/anaconda/3

# Execute the Python script within the PREFIX_CPU folder
python3 "train.py" 
