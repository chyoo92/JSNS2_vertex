#!/bain/bash

# python train_vertex_mc2.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_512_all_e4 --device 1 --epoch 10000 --batch 512 --lr 1e-4 --seed 12345 --model DGCNN6_3 --fea 1 --cla 3 --geo 1 --dtype 12

# python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_512_all_e4 --device 1 --seed 12345 --batch 1 --geo 1 --dtype 13


python train_vertex_mc2.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_4_10000_512_all_e4 --device 1 --epoch 10000 --batch 512 --lr 1e-4 --seed 12345 --model DGCNN6_4 --fea 1 --cla 3 --geo 1 --dtype 12

python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_4_10000_512_all_e4 --device 1 --seed 12345 --batch 1 --geo 1 --dtype 13

