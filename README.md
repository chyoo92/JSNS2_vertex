# JSNS2 Vertex Reconstruction with GNN 
## Run
### waveform input

- --config : config file (ex. config.yaml)
- -o : output folder name (ex. test)
- --device : choice gpu or cpu (ex. 0 or 1 or 2 ....  / cpu : -1)
- --epoch : train epoch num
- --batch : train batch size
- --lr : learning rate
- --seed : data split random number
- --model : train model
- --fea : input data feature num (ex. waveform : 248 / charge : 1)
- --cla : output feature num (ex. classification - 1 / regression(vertex) - 3)

- ex1) python train_script.py --config config.yaml -o test --device 0 --epoch 200 --batch 64 --lr 1e-4 --seed 12345 --model model --fea 248 --cla 3

- python eval_script.py --config config.yaml -o test --device 0 --seed 12345 --batch 1



### python folder




#### vertexdataset all same. it just too lazy to create input conditions.....
- python/model : GNN models
- python/dataset/vertexdataset.py : use highwf dataset
- python/dataset/vertexdataset2.py : use lowwf dataset
- python/dataset/vertexdataset3.py : use sumwf dataset



vertexdataset.py
vertexdataset2.py
vertexdataset3.py
vertexdataset_cf_ch.py
vertexdataset_cf_data.py
vertexdataset_cf_wf.py
vertexdataset_mc.py
vertexdataset_mc10.py
vertexdataset_mc2.py
vertexdataset_mc3.py
vertexdataset_mc4.py
vertexdataset_mc5.py
vertexdataset_mc6.py
vertexdataset_mc7.py
vertexdataset_mc8.py
vertexdataset_mc9.py
vertexdataset_mc_h5.py
vertexdataset_mc_h5_2.py
vertexdataset_mc_h5_3.py
vertexdataset_real_data.py



acc_loss-Copy1.ipynb
acc_loss.ipynb
Cf_new_vertex.ipynb
ch_1563_make.ipynb
charge_csv.ipynb
charge_h5.ipynb
checking_filelist.ipynb
find_Cf.ipynb
make_vertex_cf_data.ipynb
make_vertex_cf_data2.ipynb
make_vertex_proto.ipynb
make_vertex_proto2.ipynb
plot-Copy1.ipynb
plot.ipynb
plot_jade.ipynb
plot_jade_energy-Copy1.ipynb
plot_jade_energy.ipynb
Untitled.ipynb
ch_1563_make.py
ch_make_loop.sh
ch_make_loop2.sh
config_1563.yaml
config_1563_2.yaml
config_cf.yaml
config_cf_10k.yaml
config_cf_2.yaml
config_cf_selec.yaml
config_cf_selec_1511.yaml
config_cf_selec_1517.yaml
config_cf_selec_1520.yaml
config_cylinder10.yaml
config_cylinder20.yaml
config_cylinder30.yaml
config_cylinder40.yaml
config_cylinder50.yaml
config_jsns2_10.yaml
config_jsns2_20.yaml
config_jsns2_30.yaml
config_jsns2_40.yaml
config_jsns2_50.yaml
config_jsns2_all.yaml
config_jsns2_positron_10.yaml
config_jsns2_positron_1t20.yaml
config_jsns2_positron_all.yaml
config_mc.yaml
config_mc2.yaml
config_sphere10.yaml
config_sphere20.yaml
config_sphere30.yaml
config_sphere40.yaml
config_sphere50.yaml
config_test.yaml
cylinder_geometry_pos.csv
eval_vertex_cf_ch.py
eval_vertex_cf_wf.py
eval_vertex_mc.py
eval_vertex_mc2.py
eval_vertex_mc3.py
eval_vertex_mc4.py
eval_vertex_t1.py
eval_vertex_t2.py
eval_vertex_t3.py
geometry_pos.txt
jsns_geometry_pos.csv
jsns_geometry_pos2.csv
make_vertex.py
make_vertex_cf_data.py
noselec_mc.csv
noselec_mc_total1.h5
noselec_mc_total1_v2.h5
noselec_mc_total1_v3.h5
noselec_mc_total2.h5
noselec_mc_total2_v2.h5
noselec_mc_total2_v3.h5
noselec_mc_v2.csv
README.md
selec_mc.csv
selec_mc_total.h5
selec_mc_total_v2.h5
selec_mc_total_v3.h5
selec_mc_v2.csv
sphere_geometry_pos.csv
testsss.txt
train_charge1.sh
train_charge11.sh
train_charge2.sh
train_charge3.sh
train_charge4.sh
train_vertex_cf_ch.py
train_vertex_cf_wf.py
train_vertex_mc.py
train_vertex_mc2.py
train_vertex_mc2_for_multy.py
train_vertex_mc3.py
train_vertex_t1.py
train_vertex_t2.py
train_vertex_t3.py
train_wf.sh
vertex_finder.tar


### config file

- config_test.yaml : data path / split fraction / etc....


### Make file

- make_vertex.py : make vertex data from raw and cm file. for only JSNS2

### Training & Validation & Evaluation


- train_vertex_t1.py : train highwf
- train_vertex_t2.py : train lowwf
- train_vertex_t3.py : train sumwf

- eval_vertex_t1.py : eval highwf
- eval_vertex_t2.py : eval lowwf
- eval_vertex_t3.py : eval sumwf


### ipynb script

- acc_loss.ipynb : draw several results acc & loss
- checking_filelist.ipynb : cp from kek check list
- make_vertex_protp.ipynb : same "make_vertex.py", testing script
- plot.ipynb : vertex result confirm
- train_vertex_t1.ipynb : train code test script
- Untitiled.ipynb : testing script....
