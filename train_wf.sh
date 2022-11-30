#!/bain/bash
python train_vertex_t1.py --config config_jsns2_positron_1t20.yaml -o 20221130_code_clean --device 0 --epoch 1 --batch 64 --lr 1e-4 --seed 12345 --model DGCNN6_homo --fea 1 --cla 3 --geo 1 --itype 0 --ftype 1

# python eval_vertex_t1.py --config config_test.yaml -o 20221130_code_clean --device 3 --seed 12345 --batch 1 --geo 1 --itype 3  --ftype 1
