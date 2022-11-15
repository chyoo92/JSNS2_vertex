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



   
        
class vertexdataset_mc9(PyGDataset):
    def __init__(self, **kwargs):
        super(vertexdataset_mc9, self).__init__(None, transform=None, pre_transform=None)
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


        
        vertex = torch.Tensor(self.vtxList[fileIdx][idx])
        pos = torch.Tensor(self.posList[fileIdx][idx])
        charges = torch.Tensor(self.featureList[fileIdx][idx])
  
 
        data = PyGData(x = charges, pos = pos, y = vertex)
      
   

        return data
    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
            if not fName.endswith(".csv"): continue
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
    def initialize(self,geo):
        if self.isLoaded: return
    
        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())


  
        self.featureList = []
        
        self.vtxList = []
        self.posList = []
        self.procList = []

      
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):

            f = pd.read_csv(fName)

            nEvent = len(np.array(f))
             
            self.sampleInfo.loc[i, 'nEvent'] = nEvent
            if geo == 1:
                pos_file = 'jsns_geometry_pos.csv'
            elif geo == 2:
                pos_file = 'sphere_geometry_pos.csv'
            elif geo == 3:
                pos_file = 'cylinder_geometry_pos.csv'
            elif geo == 0:
                pos_file = 'jsns_geometry_pos2.csv'
    
            ff = pd.read_csv(pos_file,header=0)
#             pos = []
       
            
#             for j in range(nEvent):

#                 pos.append(np.array(ff))
            
    
#             self.featureList.append(np.concatenate((np.array(f)[:,17:17+len(ff)].reshape(-1,len(ff),1),np.array(pos).reshape(-1,len(ff),3)),axis=2).tolist())
            pos = []

            feature = []
            pmt_charge = np.array(f)[:,17:17+len(ff)]
            for j in range(nEvent):

                s = pmt_charge[j].argsort()
           
                new_charge = pmt_charge[j][s]/pmt_charge[j].sum()
            
                new_pos = np.array(ff)[s]/1000
      
                new2_charge = np.concatenate([new_charge[:45],new_charge[-45:]])
                new2_pos = np.concatenate([new_pos[:45],new_pos[-45:]])
          
                pos.append(new2_pos)
                feature.append(new2_charge.reshape(-1,1))
                
            self.featureList.append(feature)    
            self.vtxList.append(np.array(f)[:,10:13].tolist())
            self.posList.append(pos)
            

            procIdx = procNames.index(self.sampleInfo['procName'][i])
            self.procList.append(torch.ones(nEvent, dtype=torch.int32, requires_grad=False)*procIdx)
        ## Compute cumulative sums of nEvent, to be used for the file indexing
        self.maxEventsList = np.concatenate(([0.], np.cumsum(self.sampleInfo['nEvent'])))

        ## Find rescale factors - make average weight to be 1 for each cat in the training step
        for fileIdx in self.sampleInfo['fileIdx']:
            label = self.sampleInfo.loc[self.sampleInfo.fileIdx==fileIdx, 'label']
   
            for l in label: ## this loop runs only once, by construction.

                break ## this loop runs only once, by construction. this break is just for a confirmation
                print('-'*80)
        self.isLoaded = True
