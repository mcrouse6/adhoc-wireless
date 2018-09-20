import os, sys


## increasing noise scales batch run
noise_scales = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]
ref_scale = 1.0

for ns in noise_scales:
    base_str = "python train.py \
                    --epoch 50 \
                    --batch-size 128 \
                    --lr .1 \
                    --output models/noise_scales/ \
                    --quantization 0 \
                    --numreferencenodes 2 \
                    --rejectionsampling 0 \
                    --noise-scale {} \
                    --ref-scale {} \
                    --train-size 5000".format(ns, ref_scale)
    print(base_str)
    os.system(base_str)

import os, sys
## num references batch run
num_references = [1, 2, 3, 4, 5, 10]
noise_scale = 0.01
for nf in num_references:
    base_str = "python train.py \
                    --epoch 50 \
                    --batch-size 128 \
                    --lr .1 \
                    --output models/orientation_num_refs_noise_01/ \
                    --quantization 0 \
                    --numreferencenodes {} \
                    --noise-scale {} \
                    --train-size 5000".format(nf, noise_scale)
    print(base_str)
    os.system(base_str)


import os, sys
## num training examples batch run
noise_scales = [0.0, 0.01, 0.02, 0.03, 0.05, 0.1]
num_refs = [2, 3, 5, 10, 20, 30]
for nf in num_refs:
    for ns in noise_scales:
        base_str = "python train.py \
                        --epoch 50 \
                        --batch-size 128 \
                        --lr .1 \
                        --output models/noise_refs_{}/ \
                        --quantization 0 \
                        --numreferencenodes {} \
                        --rejectionsampling 0 \
                        --noise-scale {} \
                        --ref-scale 1.0 \
                        --train-size 5000".format(nf, nf, ns)
        print(base_str)
        os.system(base_str)

import os, sys
import numpy as np
## num training examples batch run
sisr_thres = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, np.inf]
num_refs = [2, 3, 5, 20, 30]
num_refs = [3]
for nf in num_refs:
    for st in sisr_thres:
        base_str = "python train.py \
                        --epoch 50 \
                        --batch-size 128 \
                        --lr .1 \
                        --output models/sisr_thres_numref_3_.2_{}/ \
                        --quantization 0 \
                        --numreferencenodes {} \
                        --rejectionsampling 0 \
                        --noise-scale .2 \
                        --ref-scale 1.0 \
                        --sisr-thres {} \
                        --train-size 5000".format(nf, nf, st)
        print(base_str)
        os.system(base_str)