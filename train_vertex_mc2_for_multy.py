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

sys.path.append("./python")
from model.allModel import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
# parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--epoch', action='store', type=int, default=400,help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=32, help='Batch size')
parser.add_argument('--lr', action='store', type=float, default=1e-4,help='Learning rate')
parser.add_argument('--seed', action='store', type=int, default=12345,help='random seed')
parser.add_argument('--fea', action='store', type=int, default=248, help='# fea')
parser.add_argument('--cla', action='store', type=int, default=3, help='# class')
parser.add_argument('--geo', action='store', type=int, default=1, help='geometry')
parser.add_argument('--dtype', action='store', type=int, default=1, help='dataset type')

 
models = ['GNN1layer', 'GNN2layer','GNN3layer', 'GNN4layer','GNN10layer','GNN11layer','GNN12layer','GNN13layer','GNN22layer','GNN33layer','GNN44layer','GNN55layer','GNN1010layer','GNN5layer','DGCNN','DGCNN2','DGCNN3','DGCNN4','DGCNN5','DGCNN6','DGCNN7','DGCNN8','DGCNN9','DGCNN10','DGCNN11','DGCNN6_2','DGCNN6_3','DGCNN6_4','GNN963layer']
parser.add_argument('--model', choices=models, default=models[0], help='model name')


args = parser.parse_args()
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
config['training']['learningRate'] = float(config['training']['learningRate'])
if args.seed: config['training']['randomSeed1'] = args.seed
if args.epoch: config['training']['epoch'] = args.epoch
if args.lr: config['training']['learningRate'] = args.lr


torch.set_num_threads(os.cpu_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:',device)
print('Current cuda device:', torch.cuda.current_device())
print('Count of using GPUs:', torch.cuda.device_count())


if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)



import time
start = time.time()
##### Define dataset instance #####
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
    
    
for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'], weight=sampleInfo['xsec']/sampleInfo['ngen'])
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize(args.geo)


lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)


kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':False}

trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())


##### Define model instance #####
exec('model = '+args.model+'(fea=args.fea, cla=args.cla)')
torch.save(model, os.path.join('result/' + args.output, 'model.pth'))


_model = model.cuda()
model = nn.DataParallel(_model).to(device)


##### Define optimizer instance #####
optm = optim.Adam(model.parameters(), lr=config['training']['learningRate'])



##### Start training #####
with open('result/' + args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()
    
    
    
from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'val_loss':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    optm.zero_grad()

    
    for i, data in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        data = data.to(device)
        
       
        label = data.y.float().to(device=device) ### vertex

        label = label.reshape(-1,3)

        pred = model(data)
    
        crit = torch.nn.MSELoss() ### sacledweight np.abs()

        loss = crit(pred, label)
        loss.backward()

        optm.step()
        optm.zero_grad()


        ibatch = len(label)
        nProcessed += ibatch
        trn_loss += loss.item()*ibatch

        
        
    trn_loss /= nProcessed 

    print(trn_loss,'trn_loss')

    model.eval()
    val_loss, val_acc = 0., 0.
    nProcessed = 0
    for i, data in enumerate(tqdm(valLoader)):
        
        data = data.to(device)

        
        label = data.y.float().to(device=device)
        label = label.reshape(-1,3)

        pred = model(data)
  
        crit = torch.nn.MSELoss()
        loss = crit(pred, label)

        
        ibatch = len(label)
        nProcessed += ibatch
        val_loss += loss.item()*ibatch

            
            
    val_loss /= nProcessed
    print(val_loss,'val_loss')
    if bestLoss > val_loss:
        bestState = model.to('cpu').state_dict()
        bestLoss = val_loss
        torch.save(bestState, os.path.join('result/' + args.output, 'weight.pth'))

        model.to(device)

    train['loss'].append(trn_loss)
    train['val_loss'].append(val_loss)

    with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)

bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))







