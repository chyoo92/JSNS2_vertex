#!/bain/bash



##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'


python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl3_k10_0_aggr_max --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type5 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 3 --edge 10 --depths 5 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl3_k10_0_aggr_max --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3

python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl1_k10_0_aggr_max --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type5 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 1 --edge 10 --depths 5 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl1_k10_0_aggr_max --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 1

python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl4_k10_0_aggr_max --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type5 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 4 --edge 10 --depths 5 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl4_k10_0_aggr_max --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 4




python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl3_k10_0_aggr_mean --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type5 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr mean --cla 3 --edge 10 --depths 5 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl3_k10_0_aggr_mean --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3

python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl1_k10_0_aggr_mean --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type5 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr mean --cla 1 --edge 10 --depths 5 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl1_k10_0_aggr_mean --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 1

python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl4_k10_0_aggr_mean --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type5 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr mean --cla 4 --edge 10 --depths 5 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230228_type5_mse_loss_cl4_k10_0_aggr_mean --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 4
