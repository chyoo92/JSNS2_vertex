#!/bain/bash



# python train_vertex_mc2.py --config config_jsns2_positron_all.yaml -o 20221122_homo_test2_all --device 2 --epoch 10000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN6_homo2 --fea 1 --cla 3 --geo 1 --dtype 12

python eval_vertex_mc3.py --config config_cf_selec_1520.yaml -o 20221122_homo_test_eval_cf_1520 --device 3 --seed 12345 --batch 1 --geo 1 --dtype 14
