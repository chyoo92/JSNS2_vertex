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
from loss_functions import *
sys.path.append("./python")
from model.allModel import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', action='store', type=str, help='Configration file with sample information')
parser.add_argument('-o', '--output', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')
parser.add_argument('--epoch', action='store', type=int, default=400,help='Number of epochs')
parser.add_argument('--batch', action='store', type=int, default=32, help='Batch size')
parser.add_argument('--lr', action='store', type=float, default=1e-4,help='Learning rate')
parser.add_argument('--seed', action='store', type=int, default=12345,help='random seed')

parser.add_argument('--fea', action='store', type=int, default=4, help='# fea')
parser.add_argument('--cla', action='store', type=int, default=3, help='# class')
parser.add_argument('--hidden', action='store', type=int, default=32, help='hidden dim num')
parser.add_argument('--heads', action='store', type=int, default=8, help='head num')
parser.add_argument('--posfeed', action='store', type=int, default=64, help='hidden dim num')
parser.add_argument('--dropout', action='store', type=float, default=1e-1,help='dropout rate')

parser.add_argument('--geo', action='store', type=int, default=1, help='geometry')
parser.add_argument('--itype', action='store', type=int, default=0, help='input data type 0 = charge, 1 = wfhigh, 2 = wflow, 3 = wf sum')
parser.add_argument('--tev', action='store', type=int, default=1, help='sample info saving 1 = training , 0 = evaluation')

parser.add_argument('--loss', action='store', type=str, default='mse', help='mse, mae, logcosh, maxcut')
parser.add_argument('--depths', action='store', type=int, default=3, help='encoder num')


models = ['SAVER']
parser.add_argument('--model', choices=models, default=models[0], help='model name')
                                                                                                                                                                                                                                                                                                                                                                                                                         

args = parser.parse_args()
config = yaml.load(open(args.config).read(), Loader=yaml.FullLoader)
config['training']['learningRate'] = float(config['training']['learningRate'])
if args.seed: config['training']['randomSeed1'] = args.seed
if args.epoch: config['training']['epoch'] = args.epoch
if args.lr: config['training']['learningRate'] = args.lr


torch.set_num_threads(os.cpu_count())
if torch.cuda.is_available() and args.device >= 0: torch.cuda.set_device(args.device)
if not os.path.exists('result/' + args.output): os.makedirs('result/' + args.output)




##### Define dataset instance #####
from dataset.vertexdataset import *
dset = vertexdataset()


for sampleInfo in config['samples']:
    if 'ignore' in sampleInfo and sampleInfo['ignore']: continue
    name = sampleInfo['name']
    dset.addSample(name, sampleInfo['path'], weight=1)
    dset.setProcessLabel(name, sampleInfo['label'])
dset.initialize(args.geo, args.itype, args.tev, args.output)


lengths = [int(x*len(dset)) for x in config['training']['splitFractions']]
lengths.append(len(dset)-sum(lengths))
torch.manual_seed(config['training']['randomSeed1'])
trnDset, valDset, testDset = torch.utils.data.random_split(dset, lengths)


kwargs = {'num_workers':min(config['training']['nDataLoaders'],os.cpu_count()), 'pin_memory':False}

trnLoader = DataLoader(trnDset, batch_size=args.batch, shuffle=True, **kwargs)
valLoader = DataLoader(valDset, batch_size=args.batch, shuffle=False, **kwargs)
torch.manual_seed(torch.initial_seed())


##### Define model instance #####
exec('model = '+args.model+'(fea = args.fea, cla=args.cla, depths = args.depths, hidden = args.hidden, heads = args.heads, posfeed = args.posfeed, dropout = args.dropout, device= args.device,batch = args.batch)')
torch.save(model, os.path.join('result/' + args.output, 'model.pth'))



device = 'cpu'
if args.device >= 0 and torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

##### Define optimizer instance #####
# optm = optim.Adam(model.parameters(), lr=config['training']['learningRate'])

# optm = optim.AdamW(model.parameters(), lr=config['training']['learningRate'])


##### Start training #####
with open('result/' + args.output+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()
    
    
if args.loss =='weight':  
    w0 = torch.tensor(0.5, requires_grad=True)
    w1 = torch.tensor(0.3, requires_grad=True)
    w2 = torch.tensor(0.2, requires_grad=True)

    def weighted_loss(loss0, loss1, loss2, w0, w1, w2):
        # 손실 함수에 가중치를 적용하여 계산
        weighted_loss = w0 * loss0 + w1 * loss1 + w2 * loss2 + (w0/w1) + (w1/w2) + (w2/w0)
        return weighted_loss, w0, w1, w2

    # w0_normalized, w1_normalized, w2_normalized = normalize_weights(w0, w1, w2)

    optm = optim.Adam([{'params': model.parameters()}, {'params': [w0,w1,w2]}], lr=config['training']['learningRate'])


elif args.loss =='weight2':  
    w0 = torch.tensor(0.5, requires_grad=True)
    w1 = torch.tensor(0.3, requires_grad=True)
    w2 = torch.tensor(0.2, requires_grad=True)

    def weighted_loss2(loss00, loss0, loss1, loss2, w0, w1, w2):
        # 손실 함수에 가중치를 적용하여 계산
        weighted_loss =loss00 + w0 * loss0 + w1 * loss1 + w2 * loss2 + (w0/w1) + (w1/w2) + (w2/w0)
        return weighted_loss, w0, w1, w2

    # w0_normalized, w1_normalized, w2_normalized = normalize_weights(w0, w1, w2)

    optm = optim.Adam([{'params': model.parameters()}, {'params': [w0,w1,w2]}], lr=config['training']['learningRate'])


else:
    optm = optim.Adam({'params': model.parameters()}, lr=config['training']['learningRate'])


from sklearn.metrics import accuracy_score
from tqdm import tqdm
bestState, bestLoss = {}, 1e9
train = {'loss':[], 'val_loss':[]}
wweight = {'w00':[],'w01':[],'w02':[],
           'w10':[],'w11':[],'w12':[]}
nEpoch = config['training']['epoch']
for epoch in range(nEpoch):
    model.train()
    trn_loss, trn_acc = 0., 0.
    nProcessed = 0
    optm.zero_grad()

    
    for i, data in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, nEpoch))):
        data = data.to(device)
        labels = data.y.float().to(device=device) ### vertex
        j_energy = data.jae.float().to(device=device)
        energy = data.tre.float().to(device=device)    
        
        if args.cla == 3:
            label = labels.reshape(-1,3)

        elif args.cla == 4:
            labels = labels.reshape(-1,3)
            energys = energy.reshape(-1,1)
            label = torch.cat([labels,energys],dim=1)
        elif args.cla ==1 :
            label = energy.reshape(-1,1)


        pred = model(data)

        if args.loss == 'mse':
            crit = torch.nn.MSELoss()
        elif args.loss == 'mae':
            crit = torch.nn.L1Loss()
        elif args.loss == 'logcosh':
            crit = LogCoshLoss()
        elif args.loss == 'maxabs':
            crit = MaxABSLoss()

        elif args.loss == 'msedistance':
            crit = Msedistance()
        elif args.loss == 'eculidean':
            crit = EuclideanDistanceLoss()
        elif args.loss =='weight':
            crit = torch.nn.MSELoss()
        elif args.loss =='weight2':
            crit = torch.nn.MSELoss()
            
        if args.cla == 4:
            crit1 = torch.nn.MSELoss()
            crit2 = LogCoshLoss()
            loss1 = crit1(pred[:,:3],label[:,:3])
            loss2 = crit2(pred[:,3],label[:,3])
            loss = loss1 + loss2
        elif args.loss == 'weight':
            loss0 = crit(pred[:,0],label[:,0])
            loss1 = crit(pred[:,1],label[:,1])
            loss2 = crit(pred[:,2],label[:,2])
            
#             w0_normalized, w1_normalized, w2_normalized = normalize_weights(w0, w1, w2)
            optm.zero_grad()
#             loss = weighted_loss(loss0, loss1, loss2, w0_normalized, w1_normalized, w2_normalized)
            loss, ww0, ww1, ww2 = weighted_loss(loss0, loss1, loss2, w0, w1, w2)

            loss.backward(retain_graph=True)
            optm.step()
        
        elif args.loss == 'weight2':
            loss0 = crit(pred[:,0],label[:,0])
            loss1 = crit(pred[:,1],label[:,1])
            loss2 = crit(pred[:,2],label[:,2])
            loss00 = crit(pred, label)
#             w0_normalized, w1_normalized, w2_normalized = normalize_weights(w0, w1, w2)
            optm.zero_grad()
#             loss = weighted_loss(loss0, loss1, loss2, w0_normalized, w1_normalized, w2_normalized)
            loss, ww00, ww01, ww02 = weighted_loss2(loss00, loss0, loss1, loss2, w0, w1, w2)

            loss.backward(retain_graph=True)
            optm.step()
        else:
            loss = crit(pred, label)
        
####기존
#         loss.backward()
            
#         optm.step()
#         optm.zero_grad()


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
        labels = data.y.float().to(device=device)
        j_energy = data.jae.float().to(device=device)
        energy = data.tre.float().to(device=device)

        if args.cla == 3:
            label = labels.reshape(-1,3)
        elif args.cla == 4:
            labels = labels.reshape(-1,3)
            energys = energy.reshape(-1,1)
            label = torch.cat([labels,energys],dim=1)
        elif args.cla ==1 :
            label = energy.reshape(-1,1)

        pred = model(data)

        if args.loss == 'mse':
            crit = torch.nn.MSELoss()
        elif args.loss == 'mae':
            crit = torch.nn.L1Loss()
        elif args.loss == 'logcosh':
            crit = LogCoshLoss()
        elif args.loss == 'maxabs':
            crit = MaxABSLoss()

        elif args.loss == 'msedistance':
            crit = Msedistance()
        elif args.loss == 'eculidean':
            crit = EuclideanDistanceLoss()
        elif args.loss =='weight':
            crit = torch.nn.MSELoss()
        elif args.loss =='weight2':
            crit = torch.nn.MSELoss()
            
        if args.loss == 'weight':
            loss0 = crit(pred[:,0],label[:,0])
            loss1 = crit(pred[:,1],label[:,1])
            loss2 = crit(pred[:,2],label[:,2])

#             w1_normalized, w2_normalized, w3_normalized = normalize_weights(w0, w1, w2)

    # 가중치가 적용된 손실 계산
#             loss = weighted_loss(loss0, loss1, loss2, w0_normalized, w1_normalized, w2_normalized)
            loss, ww10, ww11, ww12 = weighted_loss(loss0, loss1, loss2, w0, w1, w2)
        elif args.loss == 'weight2':
            loss0 = crit(pred[:,0],label[:,0])
            loss1 = crit(pred[:,1],label[:,1])
            loss2 = crit(pred[:,2],label[:,2])
            loss00 = crit(pred, label)
#             w0_normalized, w1_normalized, w2_normalized = normalize_weights(w0, w1, w2)
            optm.zero_grad()
#             loss = weighted_loss(loss0, loss1, loss2, w0_normalized, w1_normalized, w2_normalized)
            loss, ww10, ww11, ww12 = weighted_loss2(loss00, loss0, loss1, loss2, w0, w1, w2)

            loss.backward(retain_graph=True)
            optm.step()         
        
        
        else:
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

    wweight['w00'].append(ww00)
    wweight['w01'].append(ww01)
    wweight['w02'].append(ww02)
    wweight['w10'].append(ww10)
    wweight['w11'].append(ww11)
    wweight['w12'].append(ww12)
    
    
    train['loss'].append(trn_loss)
    train['val_loss'].append(val_loss)

    with open(os.path.join('result/' + args.output, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = train.keys()
        writer.writerow(keys)
        for row in zip(*[train[key] for key in keys]):
            writer.writerow(row)
    
    
    with open(os.path.join('result/' + args.output, 'balance.csv'), 'w') as f:
        writer = csv.writer(f)
        keys = wweight.keys()
        writer.writerow(keys)
        for row in zip(*[wweight[key] for key in keys]):
            writer.writerow(row)

bestState = model.to('cpu').state_dict()
torch.save(bestState, os.path.join('result/' + args.output, 'weightFinal.pth'))
