Adhoc wireless beam finding via reference devices.

Basic MLP with accurate AOA measurements (version 0.1)


Start training:

(OLD)
python train.py --epoch 50 --batch-size 128 --lr .1 --output models/model_mlp_l1_ref.pth

(With Noise and Num References)
python train.py --epoch 50 --batch-size 128 --lr .1 --output models/orientation --numreferencenodes 5 --train-size 5000 --noise-scale .05