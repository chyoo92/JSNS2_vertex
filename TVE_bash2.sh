#!/bain/bash



### fea 1 global mean pool
# for layers in {2..5..1}
# do
for classfy in {3..4}
do
for cluster in {5..15..5}
do
## aggr max
python train_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggrmax_ly5_cl${classfy}_k${cluster} --device 1 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla ${classfy} --edge ${cluster} --depths 5 

python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggrmax_ly5_cl${classfy}_k${cluster} --device 1 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}
## aggr mean
python train_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggrmean_ly5_cl${classfy}_k${cluster} --device 1 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr mean --cla ${classfy} --edge ${cluster} --depths 5 

python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggrmean_ly5_cl${classfy}_k${cluster} --device 1 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}
## aggr add
python train_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggradd_ly5_cl${classfy}_k${cluster} --device 1 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr add --cla ${classfy} --edge ${cluster} --depths 5 

python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggradd_ly5_cl${classfy}_k${cluster} --device 1 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}
done
done
# done