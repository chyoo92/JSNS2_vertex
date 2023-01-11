#!/bain/bash



##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'

python train_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml -o 20230104_mse_sum3 --device 2 --epoch 3000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN6_homo4 --fea 2 --cla 3 --geo 1 --itype 0 --ftype 1

python eval_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml  -o 20230104_mse_sum3 --device 2 --seed 12345 --batch 1 --geo 1 --itype 0  --ftype 1 --cla 3
