#!/usr/bin/env python
# coding: utf-8
# %%
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import InMemoryDataset as PyGDataset, Data as PyGData
from bisect import bisect_right
from glob import glob
import numpy as np
import math

from torch_geometric.data import Data

class vertexdataset_pt(PyGDataset):
    def __init__(self, **kwargs):
        super(vertexdataset_pt, self).__init__(None, transform=None, pre_transform=None)
        self.isLoaded = False

        self.fNames = []
        self.sampleInfo = pd.DataFrame(columns=["procName", "fileName", "weight", "label", "fileIdx","sumweight"])

    def len(self):
        return int(self.maxEventsList[-1])

    def get(self, idx):
        if not self.isLoaded: self.initialize()

        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]
        idx = int(idx - offset)



        data = self.dataList[fileIdx][idx]
         

        return data


    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
#             if not fName.endswith(".h5"): continue
            fileIdx = len(self.fNames)
            self.fNames.append(fName)

            info = {
                'procName':procName, 'weight':weight, 'nEvent':0,
                'label':0, ## default label, to be filled later
                'fileName':fName, 'fileIdx':fileIdx, 'sumweight':0,
            }
            self.sampleInfo = self.sampleInfo.append(info, ignore_index=True)
            
            
    def setProcessLabel(self, procName, label):
        self.sampleInfo.loc[self.sampleInfo.procName==procName, 'label'] = label
    def initialize(self, itype, tev, output):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())

        self.procList, self.dataList = [], []
                
        #### file num check
        nFiles = len(self.sampleInfo)
        #### Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            
            
            ### file load and check event num
            f = torch.load(fName)
            
            nEvents = len(f)
            self.sampleInfo.loc[i, 'nEvent'] = nEvents
            data_list = []
            for j in range(nEvents):

                data = Data(x = torch.reshape(torch.Tensor(f[j].x),(-1,1)), pos = torch.Tensor(f[j].pos), y = torch.Tensor(f[j].y))
                
                data.E = torch.Tensor(f[j].E)
                
  
                data.jade_vertex = torch.Tensor(f[j].jade_vertex)
                data.jade_E = torch.Tensor(f[j].jade_E)
                data_list.append(data)

            self.dataList.append(data_list)

                
                
        #### save sampleInfo file in train result path
        SI = self.sampleInfo
        if tev == 1:
            SI.to_csv('result/'+output + '/training_sampleInfo.csv')
        else:
            SI.to_csv('result/'+output + '/evaluation_sampleInfo.csv')
        
        procIdx = procNames.index(self.sampleInfo['procName'][i])
        self.procList.append(torch.ones(nEvents, dtype=torch.int32, requires_grad=False)*procIdx)
        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']

            for l in label: ## this loop runs only once, by construction.

                break ## this loop runs only once, by construction. this break is just for a confirmation
                print('-'*80)
        self.isLoaded = True

