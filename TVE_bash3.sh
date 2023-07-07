#!/bain/bash


##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'


# python train_model_tf.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed1234 --device 1 --epoch 4000 --batch 256 --lr 1e-4 --seed 1234 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss weight --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4


# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed1234 --device 1 --seed 1234 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


# python train_model_tf.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed2345 --device 1 --epoch 4000 --batch 256 --lr 1e-4 --seed 2345 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss weight --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4


# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed2345 --device 1 --seed 2345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


python train_model_tf.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed7345 --device 1 --epoch 4000 --batch 256 --lr 1e-4 --seed 7345 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss weight --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4


python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed7345 --device 1 --seed 7345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


python train_model_tf.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed10345 --device 1 --epoch 4000 --batch 256 --lr 1e-4 --seed 10345 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss weight --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4


python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_SAVER_weight_loss_seed10345 --device 1 --seed 10345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3


