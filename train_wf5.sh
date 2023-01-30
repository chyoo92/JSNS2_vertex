#!/bain/bash


python eval_vertex_t1.py --config config_cf.yaml -o 20230116_fea1_maxpool_aggrmax_ly5_cl3_k5_cf --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3
### fea 2 global mean pool

# for classfy in {3..4}
# do
# for cluster in {5..15..5}
# do

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_maxpool_aggrmax_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_maxpool_aggrmean_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_maxpool_aggradd_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggrmax_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggrmean_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea1_meanpool_aggradd_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}


# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_maxpool_aggrmax_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_maxpool_aggrmean_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_maxpool_aggradd_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_meanpool_aggrmax_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_meanpool_aggrmean_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# python eval_vertex_t1.py --config config_jsns2_positron_1to10_0113data.yaml -o 20230116_fea2_meanpool_aggradd_ly2_cl${classfy}_k${cluster} --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla ${classfy}

# done
# done