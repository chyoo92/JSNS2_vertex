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



   
        
class vertexdataset_mc_h5_3(PyGDataset):
    def __init__(self, **kwargs):
        super(vertexdataset_mc_h5_3, self).__init__(None, transform=None, pre_transform=None)
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
        jvertex = torch.Tensor(self.jvtxList[fileIdx][idx])
        pos = torch.Tensor(self.posList[fileIdx][idx])
        charges = torch.Tensor(self.chargeList[fileIdx][idx])
  
        energys = torch.Tensor(self.energyList[fileIdx][idx])

        data = PyGData(x = charges, pos = pos, y = vertex)
      
   

        return data, jvertex, energys
    def addSample(self, procName, fNamePattern, weight=1, logger=None):
        if logger: logger.update(annotation='Add sample %s <= %s' % (procName, fNames))
        print(procName, fNamePattern)

        for fName in glob(fNamePattern):
            if not fName.endswith(".h5"): continue
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


  
        self.chargeList = []
        self.vtxList = []
        self.posList = []
        self.procList = []
        self.jvtxList = []
        self.energyList = []

      
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):

            f = h5py.File(fName,'r', libver='latest', swmr=True)['events']
    
            nEvent = len(f['jade_vertex'])
             
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
       
     

            charge = []
            
            vtx = []
            pos = []
            j_vtx = []
            energy = []
         

            for j in range(nEvent):
                
                
                
                charge.append(f['pmtQ'][j][0:96].reshape(-1,1))
   
             
                vtx.append(f['vertex'][j])
          
                j_vtx.append(f['jade_vertex'][j])
                pos.append(np.array(ff))
                energy.append([int(fName.split('_')[-2][:-3])])

            
       

            self.chargeList.append(charge)
            self.vtxList.append(vtx)
            self.jvtxList.append(j_vtx)
            self.posList.append(pos)
            self.energyList.append(energy)
            
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
