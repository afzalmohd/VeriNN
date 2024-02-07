#!/bin/bash
curl --location-trusted -u 22m0741:4e31608f35f5430d2a5bf3e17af8b45d "https://internet-sso.iitb.ac.in/login.php" > /dev/null
# Activate conda environment
source ~/anaconda3/bin/activate
conda activate verinn

# Set project directory and binary name
PROJECT_DIR=/home/afzal/tools/VeriNN
BIN_NAME=drefine

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PROJECT_DIR}/deep_refine/ex_tools/boost_1_68_0/installed/lib:${PROJECT_DIR}/deep_refine/ex_tools/gurobi912/linux64/lib:/home/afzal/anaconda3/envs/verinn/lib



