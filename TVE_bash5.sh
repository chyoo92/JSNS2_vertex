#!/bain/bash

python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mse_loss_cl3_k10_0_l7 --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 3 --edge 10 --depths 7 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mse_loss_cl3_k10_0_l7 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3



python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mse_loss_cl3_k10_0_l9 --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 3 --edge 10 --depths 9 --loss mse

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mse_loss_cl3_k10_0_l9 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mae_loss_cl3_k10_0_l7 --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 3 --edge 10 --depths 7 --loss mae

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mae_loss_cl3_k10_0_l7 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3



python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mae_loss_cl3_k10_0_l9 --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 3 --edge 10 --depths 9 --loss mae

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_mae_loss_cl3_k10_0_l9 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3
