#!/usr/bin/env python
import numpy as np
import torch
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as PyG
from torch_geometric.transforms import Distance
from torch_geometric.data import DataLoader
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Data
import sys, os
import subprocess
import csv, yaml
import math
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.optim as optim
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

sys.path.append("./python")

parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, default='config.yaml', help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output file')

parser.add_argument('-a', '--all', action='store_true', help='use all events for the evaluation, no split')
parser.add_argument('--cla', action='store', type=int, default=3, help='# class')
parser.add_argument('--geo', action='store', type=int, default=1, help='geometry')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('--seed', action='store', type=int, default=12345, help='random seed')
parser.add_argument('--dtype', action='store', type=int, default=1, help='dataset type')

args = parser.parse_args()

config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
if args.seed: config['training']['randomSeed1'] = args.seed

sys.path.append("./python")

torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)

if args.dtype == 1:
    from dataset.vertexdataset_mc import *
    dset = vertexdataset_mc()
elif args.dtype == 2:
    from dataset.vertexdataset_mc2 import *
    dset = vertexdataset_mc2()
elif args.dtype == 3:
    from dataset.vertexdataset_mc3 import *
    dset = vertexdataset_mc3()
elif args.dtype == 4:
    from dataset.vertexdataset_mc4 import *
    dset = vertexdataset_mc4()
elif args.dtype == 5:
    from dataset.vertexdataset_mc5 import *
    dset = vertexdataset_mc5()
elif args.dtype == 6:
    from dataset.vertexdataset_mc6 import *
    dset = vertexdataset_mc6()
elif args.dtype == 7:
    from dataset.vertexdataset_mc7 import *
    dset = vertexdataset_mc7()
elif args.dtype == 8:
    from dataset.vertexdataset_mc8 import *
    dset = vertexdataset_mc8()
elif args.dtype == 9:
    from dataset.vertexdataset_mc9 import *
    dset = vertexdataset_mc9()
elif args.dtype == 10:
    from dataset.vertexdataset_mc10 import *
    dset = vertexdataset_mc10()
elif args.dtype == 11:
    from dataset.vertexdataset_real_data import *
    dset = vertexdataset_real_data()        
elif args.dtype == 12:
    from dataset.vertexdataset_mc_h5 import *
    dset = vertexdataset_mc_h5()            
elif args.dtype == 13:
    from dataset.vertexdataset_mc_h5_2 import *
    dset = vertexdataset_mc_h5_2()      
elif args.dtype == 14:
    from dataset.vertexdataset_cf_data import *
    dset = vertexdataset_cf_data()           
    

for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize(args.geo)
lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
kwargs = {'num_workers':min(config['training']['nDataLoaders'], os.cpu_count()),
          'batch_size':args.batch, 'pin_memory':False}

if args.all:
    testLoader = DataLoader(dset, **kwargs)
else:
    trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)
    #testLoader = DataLoader(trnDset, **kwargs)
    #testLoader = DataLoader(valDset, **kwargs)
    testLoader = DataLoader(testDset, **kwargs)
torch.manual_seed(torch.initial_seed())

##### Define model instance #####
from model.allModel import *

model = torch.load('result/' + args.output+'/model.pth', map_location='cpu')
model.load_state_dict(torch.load('result/' + args.output+'/weight.pth', map_location='cpu'))
if args.cla == 1:
    model.fc.add_module('output', torch.nn.Sigmoid())


device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

dd = 'result/' + args.output + '/train.csv'

dff = pd.read_csv(dd)



##### Start evaluation #####
from tqdm import tqdm
labels, preds = [], []
weights = []
scaledWeights = []
procIdxs = []
fileIdxs = []
idxs = []
features = []
batch_size = []
real_weights = []
scales = []

eval_resampling = []
eval_real = []
model.eval()
val_loss, val_acc = 0., 0.
# for i, (data, label0, weight, rescale, procIdx, fileIdx, idx, dT, dVertex, vertexX, vertexY, vertexZ) in enumerate(tqdm(testLoader)):
for i, data in enumerate(tqdm(testLoader)):
    
    data = data.to(device)
    label = data.y.float().to(device=device) ### vertex
      
    label = label.reshape(-1,3)
    pred = model(data)

   
    labels.extend([x.item() for x in label.view(-1)])

    preds.extend([x.item() for x in pred.view(-1)])
    batch_size.append(data.x.shape[0])

df = pd.DataFrame({'prediction':preds, 'label':labels})
fPred = 'result/' + args.output + '/' + args.output + '.csv'
df.to_csv(fPred, index=False)


df2 = pd.DataFrame({'batch':batch_size})
fPred2 = 'result/' + args.output + '/' + args.output + '_batch.csv'
df2.to_csv(fPred2, index=False)

    
    
    
    
    
    
    
    
    
    
    
    
    
# if args.cla ==3:
#     df = pd.DataFrame({'label':labels, 'weight':weights, 'scaledWeight':scaledWeights})
#     fPred = 'result/' + args.output + '/' + args.output + '.csv'
#     df.to_csv(fPred, index=False)

#     df2 = pd.DataFrame({'prediction':preds})
#     predonlyFile = 'result/' + args.output + '/' + args.output + '_pred.csv'
#     df2.to_csv(predonlyFile, index=False)
# else:
#     df = pd.DataFrame({'label':labels, 'prediction':preds,
#                      'weight':weights, 'scaledWeight':scaledWeights})
#     fPred = 'result/' + args.output + '/' + args.output + '.csv'
#     df.to_csv(fPred, index=False)



