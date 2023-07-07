#!/bain/bash


##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'




# python train_model_tf.py --config config_test.yaml -o 20230622_SAVER_test2 --device 0 --epoch 1 --batch 256 --lr 1e-4 --seed 12345 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss mse --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4


# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230622_SAVER_test --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3




# python train_model_tf.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230622_SAVER_test_logcosh --device 0 --epoch 1000 --batch 256 --lr 1e-4 --seed 12345 --model SAVER --geo 1 --itype 0 --tev 1 --cla 3 --depths 4 --loss logcosh --hidden 32 --heads 8 --posfeed 64 --dropout 1e-1 --fea 4

# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230622_SAVER_test_logcosh --device 0 --seed 12345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3














# # python train_model.py --config config_test.yaml -o 20230630_weight_loss_test --device 0 --epoch 4000 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight



# # python train_model.py --config config_test.yaml -o 20230704_weight_loss_test --device 0 --epoch 4000 --batch 256 --lr 1e-4 --seed 12345 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight




# python train_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed1234 --device 0 --epoch 4000 --batch 256 --lr 1e-4 --seed 1234 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight


# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed1234 --device 0 --seed 1234 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3




# python train_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed2345 --device 0 --epoch 4000 --batch 256 --lr 1e-4 --seed 2345 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight


# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed2345 --device 0 --seed 2345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3









python train_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed7345 --device 0 --epoch 4000 --batch 256 --lr 1e-4 --seed 7345 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight



python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed7345 --device 0 --seed 7345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3



# python train_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed10345 --device 0 --epoch 4000 --batch 256 --lr 1e-4 --seed 10345 --model SGCNN_type1 --geo 1 --itype 0 --tev 1 --fea 1 --pools 1 --aggr max --cla 3 --edge 15 --depths 5 --loss weight


# python eval_model.py --config config_jsns2_positron_1to10_0324data.yaml -o 20230704_weight_loss_test_seed10345 --device 0 --seed 10345 --batch 1 --geo 1 --itype 0 --tev 0 --cla 3



