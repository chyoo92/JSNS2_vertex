#!/bain/bash



##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'

# python train_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml -o 20221211_homo4 --device 0 --epoch 10000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN6_homo4 --fea 2 --cla 3 --geo 1 --itype 0 --ftype 1

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml  -o 20221211_homo4 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0  --ftype 1


# python train_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml -o 20221211_homo4_lr5 --device 0 --epoch 10000 --batch 256 --lr 1e-5 --seed 12345 --model DGCNN6_homo4 --fea 2 --cla 3 --geo 1 --itype 0 --ftype 1

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml  -o 20221211_homo4_lr5 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0  --ftype 1


python eval_vertex_t1.py --config config_jsns2_positron_1to10_1207data.yaml  -o 20221207_jade_charge_test_homo2 --device 0 --seed 12345 --batch 1 --geo 1 --itype 0  --ftype 1