#!/bin/bash

# Just creates keypoint ymls, kp.yml
python3 sixd_dataset_all.py -d ~/DATASETS/sixd_2017

# Does all ply, kp ply, model rotation/translation and kp.yml files
# python3 sixd_dataset_all.py -d ~/DATASETS/sixd_2017 -p -r -k
