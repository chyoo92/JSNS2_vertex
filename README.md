# JSNS2 Vertex Reconstruction Deep Learning

## Install Conda
~~~
## conda version 22.9.0
## python 3.9
## pytorch 1.13.0 version
## CUDA 11.7
## install pyg 2.3.1

## Main library
conda create -n env_name python=3.9
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html

## Other library

pip install h5py

pip install matplotlib

pip install jupyter

pip install ipykernel  ## install ipykernel (for vscode)

pip install pandas==1.5.3  ## pandas >= 2.0.0 version can't use 'append' so use pandasn <=2.0.0 version

pip install uproot

conda install -c conda-forge root  ## 


############################################################################################### 
Most installations are done using pip rather than conda.
Due to the conda version issue, installation does not work properly when done with conda.
Only the root user installed using conda.
If the conda version is up to date, there should be no issues with installing using conda.
###############################################################################################
~~~

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

- python/detector_geometry/ *.csv : make it according to your detector and us it. (load from vertexdataset.py)







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




<details><summary>흐음</summary>
<p>

- python/detector_geometry/jsns_geometry_pos.csv : jsns 96 PMTs geo
- python/detector_geometry/jsns_geometry_pos2.csv : jsns 120 PMTs geo
- python/detector_geometry/cylinder_geometry_pos.csv : RAT example cylinder geo
- python/detector_geometry/sphere_geometry_pos.csv : RAT sphere cylinder geo

- python/old_config_file/*.yaml : old config file, now not use, just save

- python/old_mc_data/*.csv & *.h5 : old mc data, not use


</p>
</deteils>