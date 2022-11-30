# JSNS2 Vertex Reconstruction with GNN 
# Run
## waveform input
### Run script example
    python train_script.py --config config.yaml -o test --device # --epoch # --batch # --lr # --seed # --model model --fea # --cla # --geo # --itype # --ftype #
    python eval_script.py --config config.yaml -o test --device # --seed # --batch # --geo # --itype #  --ftype #
### condition
    --config : config file (ex. config.yaml)
    -o : output folder name (ex. test)
    --device : choice gpu or cpu (ex. 0 or 1 or 2 ....  / cpu : -1)
    --epoch : train epoch num
    --batch : train batch size
    --lr : learning rate
    --seed : data split random number
    --model : train model
    --fea : input data feature num (ex. waveform : 248 / charge : 1)
    --cla : output feature num (ex. classification - 1 / regression(vertex) - 3)
    --geo : detector geometry (0: jsns2 120 / 1 : jsns2 96 / 2 : sphere mc / 3 : cylinder mc)
    --itype : input type (input data type 0=charge, 1=wf high, 2 = wf low, 3 = wf sum)
    --ftype : file type 0 = csv / 1 = h5


### python folder

- python/model : GNN models

- python/dataset/vertexdataset.py : dataset processing script

- python/detector_geometry/jsns_geometry_pos.csv : jsns 96 PMTs geo
- python/detector_geometry/jsns_geometry_pos2.csv : jsns 120 PMTs geo
- python/detector_geometry/cylinder_geometry_pos.csv : RAT example cylinder geo
- python/detector_geometry/sphere_geometry_pos.csv : RAT sphere cylinder geo

- python/old_config_file/*.yaml : old config file, now not use, just save

- python/old_mc_data/*.csv & *.h5 : old mc data, not use







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



config_cf.yaml
config_jsns2_positron_10.yaml
config_jsns2_positron_1t20.yaml
config_jsns2_positron_all.yaml
config_test.yaml




eval_vertex_cf_ch.py
eval_vertex_cf_wf.py
eval_vertex_mc.py
eval_vertex_mc2.py
eval_vertex_mc3.py
eval_vertex_mc4.py
eval_vertex_t1.py
eval_vertex_t2.py
eval_vertex_t3.py



make_vertex.py
make_vertex_cf_data.py






README.md


train_vertex_t1.py

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
