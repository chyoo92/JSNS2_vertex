#### 4top resamping & 4top,QCD GNN classification


### python folder
#### vertexdataset all same. it just too lazy to create input conditions.....
- python/model : GNN models
- python/dataset/vertexdataset.py : use highwf dataset
- python/dataset/vertexdataset2.py : use lowwf dataset
- python/dataset/vertexdataset3.py : use sumwf dataset

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