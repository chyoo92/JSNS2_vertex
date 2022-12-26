#!/bain/bash



##'input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum'

##'file type 0 = csv / 1 = h5'


python eval_vertex_t1.py --config config_cf.yaml  -o 20221211_homo4_cf --device 0 --seed 12345 --batch 1 --geo 1 --itype 0  --ftype 1
