#!/bain/bash


# python train_vertex_mc2.py --config config_jsns2_positron_1t20.yaml -o 202211107_model_6_3_20000_2048_1to10 --device 0 --epoch 20000 --batch 2048 --lr 1e-5 --seed 12345 --model DGCNN6_3 --fea 1 --cla 3 --geo 1 --dtype 12

# python eval_vertex_mc2.py --config config_1563_2.yaml -o 202211107_model_6_3_20000_2048_1to10_1563_2 --device 0 --seed 12345 --batch 1 --geo 1 --dtype 11



# python eval_vertex_mc2.py --config config_cf.yaml -o 202211107_model_6_3_20000_2048_1to10_cf --device 0 --seed 12345 --batch 1 --geo 1 --dtype 11

python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211107_model_6_3_20000_1024_all --device 0 --seed 12345 --batch 1 --geo 1 --dtype 13