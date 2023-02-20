#!/bain/bash


python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_logcosh_loss_cl3_k10_0 --device 2 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 3 --edge 10 --depths 5 --loss logcosh

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_logcosh_loss_cl3_k10_0 --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_logcosh_loss_cl1_k10_0 --device 2 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 1 --edge 10 --depths 5 --loss logcosh

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_logcosh_loss_cl1_k10_0 --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 1


python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_logcosh_loss_cl4_k10_0 --device 2 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 0 --aggr max --cla 4 --edge 10 --depths 5 --loss logcosh 

python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230220_type1_logcosh_loss_cl4_k10_0 --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 4



# ### fea 2 global max pool
# # for layers in {2..5..1}
# # do
# # for classfy in {3..4}
# # do
# for cluster in {5..15..5}
# do
# ## aggr max
# # python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230206_fea2_maxpool_aggrmax_ly5_cl1_k${cluster} --device 2 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 2 --pools 0 --aggr max --cla 1 --edge ${cluster} --depths 5 

# python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230206_fea2_maxpool_aggrmax_ly5_cl1_k${cluster} --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 1
# ## aggr mean
# # python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230206_fea2_maxpool_aggrmean_ly5_cl1_k${cluster} --device 2 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 2 --pools 0 --aggr mean --cla 1 --edge ${cluster} --depths 5 

# python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230206_fea2_maxpool_aggrmean_ly5_cl1_k${cluster} --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 1
# ## aggr add
# # python train_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230206_fea2_maxpool_aggradd_ly5_cl1_k${cluster} --device 2 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model DGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 2 --pools 0 --aggr add --cla 1 --edge ${cluster} --depths 5 

# python eval_model.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230206_fea2_maxpool_aggradd_ly5_cl1_k${cluster} --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 1
# done
# # done
# # done