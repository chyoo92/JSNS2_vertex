#!/bain/bash


python train_vertex_mc2.py --config config_jsns2_positron_all.yaml -o 202211107_model_6_3_20000_256_all --device 0 --epoch 20000 --batch 1024 --lr 1e-5 --seed 12345 --model DGCNN6_3 --fea 1 --cla 3 --geo 1 --dtype 12

python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211107_model_6_3_20000_256_all --device 0 --seed 12345 --batch 1 --geo 1 --dtype 13

