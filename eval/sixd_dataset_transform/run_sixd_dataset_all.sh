#!/bin/bash

# Just creates keypoint ymls, kp.yml
# python3 sixd_dataset_all.py -d ~/DATASETS/linemod_plus

# Does all ply, kp ply, model rotation/translation and kp.yml files
python3 sixd_dataset_all.py -d ~/DATASETS/linemod_plus -p -r -k
