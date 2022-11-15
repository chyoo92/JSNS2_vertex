#!/bain/bash
python train_vertex_t1.py --config config_test.yaml -o 20220826_high_1L --device 3 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model GNN1layer_wf --fea 248 --cla 3

python eval_vertex_t1.py --config config_test.yaml -o 20220826_high_1L --device 3 --seed 12345 --batch 1

python train_vertex_t1.py --config config_test.yaml -o 20220826_high_2L --device 3 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model GNN2layer_wf --fea 248 --cla 3

python eval_vertex_t1.py --config config_test.yaml -o 20220826_high_2L --device 3 --seed 12345 --batch 1



python train_vertex_t1.py --config config_test.yaml -o 20220826_high_3L --device 3 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model GNN3layer_wf --fea 248 --cla 3

python eval_vertex_t1.py --config config_test.yaml -o 20220826_high_3L --device 3 --seed 12345 --batch 1

python train_vertex_t1.py --config config_test.yaml -o 20220826_high_4L --device 3 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model GNN4layer_wf --fea 248 --cla 3

python eval_vertex_t1.py --config config_test.yaml -o 20220826_high_4L --device 3 --seed 12345 --batch 1



python train_vertex_cf_wf.py --config config_cf.yaml -o 20220826_cf_wf_test1_L3 --device 3 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model GNN3layer_wf --fea 248 --cla 3

python eval_vertex_cf_wf.py --config config_cf.yaml -o 20220826_cf_wf_test1_L3 --device 3 --seed 12345 --batch 1


python train_vertex_cf_wf.py --config config_cf.yaml -o 20220826_cf_wf_test1_L4 --device 3 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model GNN4layer_wf --fea 248 --cla 3

python eval_vertex_cf_wf.py --config config_cf.yaml -o 20220808_cf_wf_test1_L4 --device 3 --seed 12345 --batch 1