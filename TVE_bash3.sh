#!/bain/bash


##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'










python train_model_pt.py --config config_jsns2_pos_charge_20230830_1000k.yaml -o 20230901_pos_charge_3l_15e_seed12345_1000k_dgcnn --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

python eval_model_pt.py --config config_jsns2_pos_charge_20230830_1000k.yaml -o 20230901_pos_charge_3l_15e_seed12345_1000k_dgcnn --device 0 --seed 12345 --batch 1 --itype 0 --tev 0 --cla 3




# python train_model_pt.py --config config_jsns2_pos_charge_20230830_1000k.yaml -o 20230830_pos_charge_3l_15e_seed12345_1000k --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230830_1000k.yaml -o 20230830_pos_charge_3l_15e_seed12345_1000k --device 0 --seed 12345 --batch 1 --itype 0 --tev 0 --cla 3






# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_15e_seed12345 --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_15e_seed12345 --device 0 --seed 12345 --batch 1 --itype 0 --tev 0 --cla 3



# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_15e_seed22345 --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 22345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_15e_seed22345 --device 0 --seed 22345 --batch 1 --itype 0 --tev 0 --cla 3



# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_15e_seed27345 --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 27345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_15e_seed27345 --device 0 --seed 27345 --batch 1 --itype 0 --tev 0 --cla 3





# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_30e_seed12345 --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 30 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_30e_seed12345 --device 0 --seed 12345 --batch 1 --itype 0 --tev 0 --cla 3



# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_30e_seed22345 --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 22345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 30 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_30e_seed22345 --device 0 --seed 22345 --batch 1 --itype 0 --tev 0 --cla 3



# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_30e_seed27345 --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 27345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 30 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2.yaml -o 20230824_pos_charge_3l_30e_seed27345 --device 0 --seed 27345 --batch 1 --itype 0 --tev 0 --cla 3










# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2_all.yaml -o 20230824_pos_charge_3l_15e_seed12345_all --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2_all.yaml -o 20230824_pos_charge_3l_15e_seed12345_all --device 0 --seed 12345 --batch 1 --itype 0 --tev 0 --cla 3






# python train_model_pt.py --config config_jsns2_pos_charge_20230811_v2_all.yaml -o 20230824_pos_charge_3l_30e_seed12345_all --device 0 --epoch 3500 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1_forpt --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 30 --depths 3 --loss logcosh

# python eval_model_pt.py --config config_jsns2_pos_charge_20230811_v2_all.yaml -o 20230824_pos_charge_3l_30e_seed12345_all --device 0 --seed 12345 --batch 1 --itype 0 --tev 0 --cla 3


