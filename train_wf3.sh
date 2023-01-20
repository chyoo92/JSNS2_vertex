#!/bain/bash

### fea 2 global max pool
for layers in {2..5..1}
do
for classfy in {3..4}
do
for cluster in {5..15..5}
do
## aggr max
python train_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_maxpool_aggrmax_ly${layers}_cl${classfy}_k${cluster} --device 2 --epoch 3000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 2 --pools 0 --aggr max --cla ${classfy} --edge ${cluster} --depths ${layers} 

python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yam -o 20230116_fea2_maxpool_aggrmax_ly${layers}_cl${classfy}_k${cluster} --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}
## aggr mean
python train_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_maxpool_aggrmean_ly${layers}_cl${classfy}_k${cluster} --device 2 --epoch 3000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 2 --pools 0 --aggr mean --cla ${classfy} --edge ${cluster} --depths ${layers} 

python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yam -o 20230116_fea2_maxpool_aggrmean_ly${layers}_cl${classfy}_k${cluster} --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}
## aggr add
python train_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_maxpool_aggradd_ly${layers}_cl${classfy}_k${cluster} --device 2 --epoch 3000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 2 --pools 0 --aggr add --cla ${classfy} --edge ${cluster} --depths ${layers} 

python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yam -o 20230116_fea2_maxpool_aggradd_ly${layers}_cl${classfy}_k${cluster} --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}
done
done
done