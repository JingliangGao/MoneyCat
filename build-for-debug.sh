#!/bin/bash

# set variables
PROJECT_PWD=$(pwd)

# accquire the lastest data
# cd ${PROJECT_PWD}
# python3 acquire_data.py

# predict the next lottery numbers
cd ${PROJECT_PWD}
python3 number_predict.py


echo "[INFO] Prediction completed!"
