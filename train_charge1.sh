#!/bain/bash


python train_vertex_mc2.py --config config_jsns2_positron_1t20.yaml -o 20221122_homo_test2 --device 0 --epoch 10000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN6_homo2 --fea 1 --cla 3 --geo 1 --dtype 12
