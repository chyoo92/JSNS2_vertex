#!/bain/bash

# python train_vertex_mc2.py --config config_jsns2_positron_1t20.yaml -o 20221122_homo_test --device 0 --epoch 10000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN6_homo --fea 1 --cla 3 --geo 1 --dtype 12

# python eval_vertex_mc4.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_512_all_e4 --device 0 --seed 12345 --batch 1 --geo 1 --dtype 15




# python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_512_all_e5 --device 0 --seed 12345 --batch 1 --geo 1 --dtype 13



# python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_512_all_e4 --device 0 --seed 12345 --batch 1 --geo 1 --dtype 13

# python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_1024_all_e5 --device 0 --seed 12345 --batch 1 --geo 1 --dtype 13


# python eval_vertex_mc3.py --config config_jsns2_positron_all.yaml -o 202211116_model_6_3_10000_1024_all_e4 --device 0 --seed 12345 --batch 1 --geo 1 --dtype 13

python eval_vertex_mc3.py --config config_cf_selec.yaml -o 20221122_homo_test_eval_cf --device 0 --seed 12345 --batch 256 --geo 1 --dtype 14
