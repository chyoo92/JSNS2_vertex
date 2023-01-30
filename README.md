# JSNS2 Vertex Reconstruction with Dynamic Graph Convolution Neural Network (DGCNN)
## Conda instll
Install pytorch-geomtric (pyg)
Additional libs may be required...


    conda create -n conda_name python=3.9

    conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=10.2 -c pytorch

    conda install pyg -c pyg

    conda install -c anaconda h5py

    conda install -c conda-forge matplotlib
## Run option
    
    --config : config file (ex. config.yaml)
    --output / -o : output folder name (ex. test)
    --device : choice gpu or cpu (ex. 0 or 1 or 2 ....  / cpu : -1)
    --epoch : train epoch num
    --batch : train batch size
    --lr : learning rate
    --seed : data split random number
    --fea : input data feature num (ex. waveform : 248 / charge : 1)
    --cla : output feature num (ex. classification - 1 / regression(vertex) - 3)
    --geo : detector geometry (for load PMTs position infor)| 1 = jsns2 96PMTs / 2 = jsns2 120PMTs
    --itype : input feature type 1,2,3(wave form high, low, sum) 0 = pmt charge
    --tev : sample info saving 1 = training , 0 = evaluation
    --edge : dgcnn layer Number of nearest neighbors
    --aggr : The aggregation operator The aggregation operator "add","mean","max"
    --depths : dgcnn Number of layers
    --pools : global pool max 0 / mean 1


    --model : train model

### Run script example
    python train_model.py --config config_name.yaml -o output --device # --epoch # --batch # --lr # --seed # --model model_name --geo # --itype # --tev 1 --fea # --pools # --aggr operator --cla # --edge # --depths # 

    python eval_model.py --config config_name.yaml -o output --device # --seed # --batch # --geo # --itype # --tev 0 --cla #


### python folder

- python/model : GNN models

- python/dataset/vertexdataset.py : dataset processing script

- python/detector_geometry/jsns_geometry_pos.csv : jsns 96 PMTs geo
- python/detector_geometry/jsns_geometry_pos2.csv : jsns 120 PMTs geo
- python/detector_geometry/cylinder_geometry_pos.csv : RAT example cylinder geo
- python/detector_geometry/sphere_geometry_pos.csv : RAT sphere cylinder geo

- python/old_config_file/*.yaml : old config file, now not use, just save

- python/old_mc_data/*.csv & *.h5 : old mc data, not use










### config file
data path / split fraction / etc....
- config_test.yaml : for train/eval code running test
- config_cf.yaml : Cf data evaluation
- config_jsns2_positron_1to10_0113data.yaml : MC 1MeV~10MeV data

### Make file

- make_vertex.py : make vertex data from raw and cm file. for only JSNS2



### ipynb script

- acc_loss.ipynb : draw several results acc & loss
- checking_filelist.ipynb : cp from kek check list
- make_vertex_protp.ipynb : same "make_vertex.py", testing script
- plot.ipynb : vertex result confirm
- train_vertex_t1.ipynb : train code test script
- Untitiled.ipynb : testing script....
