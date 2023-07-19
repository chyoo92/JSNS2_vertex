#!/bain/bash


##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'

python train_model.py --config config_jsns2_positron_1to10_0324data_811.yaml -o 20230710_weight_loss_test_weight1 --device 2 --epoch 4000 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight



python eval_model.py --config config_jsns2_positron_1to10_0324data_811.yaml -o 20230710_weight_loss_test_weight1 --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


python train_model_tf.py --config config_jsns2_positron_1to10_0324data_811.yaml -o 20230710_SAVER_weight_loss_weight1 --device 2 --epoch 4000 --batch 256 --lr 1e-4 --seed 12345 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss weight --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4


python eval_model.py --config config_jsns2_positron_1to10_0324data_811.yaml -o 20230710_SAVER_weight_loss_weight1 --device 2 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


