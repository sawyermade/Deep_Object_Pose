#!/bin/bash

# Does all ply, kp ply, model rotation/translation and kp.yml files
python3 sixd_dataset_transform.py -d ~/DATASETS/sixd_2017 -p -r -k
