#!/bain/bash


##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'


python train_model_pt.py --config config_jsns2_pos_charge_20230811.yaml -o 20230814_remake_pt_test --device 0 --epoch 2 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

