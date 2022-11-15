#!/bain/bash


python train_vertex_mc2.py --config config_jsns2_positron_1t20.yaml -o 202211107_model_6_4_20000_512_1to10 --epoch 20000 --device 3 --batch 512 --lr 1e-5 --seed 12345 --model DGCNN6_4 --fea 1 --cla 3 --geo 1 --dtype 12

python eval_vertex_mc3.py --config config_jsns2_positron_1t20.yaml -o 202211107_model_6_4_20000_512_1to10 --device 3 --seed 12345 --batch 1 --geo 1 --dtype 13

