#!/bain/bash



python train_vertex_mc2.py --config config_jsns2_positron_all.yaml -o 20221122_homo_test2_all_512 --device 3 --epoch 10000 --batch 512 --lr 1e-4 --seed 12345 --model DGCNN6_homo2 --fea 1 --cla 3 --geo 1 --dtype 12
