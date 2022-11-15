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



   
        
class vertexdataset_cf_wf(PyGDataset):
    def __init__(self, **kwargs):
        super(vertexdataset_cf_wf, self).__init__(None, transform=None, pre_transform=None)
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
        highwf = torch.Tensor(self.highList[fileIdx][idx])
        lowwf = torch.Tensor(self.lowList[fileIdx][idx])
 
        data = PyGData(x = highwf, pos = pos, y = vertex)
      
   

        return data
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
    def initialize(self):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())


        self.highList = []
        self.lowList = []
        self.vtxList = []
        self.posList = []
        self.procList = []

        
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):

       
            f = h5py.File(fName,'r', libver='latest', swmr=True)['events']
    
            nEvent = len(f['high'])
             
            self.sampleInfo.loc[i, 'nEvent'] = nEvent
            
            pmt_pos = np.array([[0,1709.18,-707.96,1200.0],[1,1126.21,-1467.7,1200.0],[2,241.47,-1834.17,1200.0],[3,-707.96,-1709.18,1200.0],
[4,-1467.7,-1126.21,1200.0],[5,-1834.17,-241.47,1200.0],[6,-1709.18,707.96,1200.0],[7,-1126.21,1467.7,1200.0],
[8,-241.48,1834.17,1200.0],[9,707.96,1709.18,1200.0],[10,1467.7,1126.21,1200.0],[11,1834.17,241.48,1200.0],
[12,1709.18,-707.96,600.0],[13,1126.21,-1467.7,600.0],[14,241.47,-1834.17,600.0],[15,-707.96,-1709.18,600.0],
[16,-1467.7,-1126.21,600.0],[17,-1834.17,-241.47,600.0],[18,-1709.18,707.96,600.0],[19,-1126.21,1467.7,600.0],
[20,-241.48,1834.17,600.0],[21,707.96,1709.18,600.0],[22,1467.7,1126.21,600.0],[23,1834.17,241.48,600.0],
[24,1709.18,-707.96,0.0],[25,1126.21,-1467.7,0.0],[26,241.47,-1834.17,0.0],[27,-707.96,-1709.18,0.0],
[28,-1467.7,-1126.21,0.0],[29,-1834.17,-241.47,0.0],[30,-1709.18,707.96,0.0],[31,-1126.21,1467.7,0.0],
[32,-241.48,1834.17,0.0],[33,707.96,1709.18,0.0],[34,1467.7,1126.21,0.0],[35,1834.17,241.48,0.0],
[36,1709.18,-707.96,-600.0],[37,1126.21,-1467.7,-600.0],[38,241.47,-1834.17,-600.0],[39,-707.96,-1709.18,-600.0],
[40,-1467.7,-1126.21,-600.0],[41,-1834.17,-241.47,-600.0],[42,-1709.18,707.96,-600.0],[43,-1126.21,1467.7,-600.0],
[44,-241.48,1834.17,-600.0],[45,707.96,1709.18,-600.0],[46,1467.7,1126.21,-600.0],[47,1834.17,241.48,-600.0],
[48,1709.18,-707.96,-1200.0],[49,1126.21,-1467.7,-1200.0],[50,241.47,-1834.17,-1200.0],[51,-707.96,-1709.18,-1200.0],
[52,-1467.7,-1126.21,-1200.0],[53,-1834.17,-241.47,-1200.0],[54,-1709.18,707.96,-1200.0],[55,-1126.21,1467.7,-1200.0],
[56,-241.48,1834.17,-1200.0],[57,707.96,1709.18,-1200.0],[58,1467.7,1126.21,-1200.0],[59,1834.17,241.48,-1200.0],
[60,550.0,-0.0,1470.0],[61,275.0,-476.31,1470.0],[62,-275.0,-476.31,1470.0],[63,-550.0,-0.0,1470.0],
[64,-275.0,476.31,1470.0],[65,275.0,476.31,1470.0],[66,1062.52,-284.7,1470.0],[67,777.82,-777.82,1470.0],
[68,284.7,-1062.52,1470.0],[69,-284.7,-1062.52,1470.0],[70,-777.82,-777.82,1470.0],[71,-1062.52,-284.7,1470.0],
[72,-1062.52,284.7,1470.0],[73,-777.82,777.82,1470.0],[74,-284.7,1062.52,1470.0],[75,284.7,1062.52,1470.0],
[76,777.82,777.82,1470.0],[77,1062.52,284.7,1470.0],[78,550.0,-0.0,-1470.0],[79,275.0,-476.31,-1470.0],
[80,-275.0,-476.31,-1470.0],[81,-550.0,-0.0,-1470.0],[82,-275.0,476.31,-1470.0],[83,275.0,476.31,-1470.0],
[84,1062.52,-284.7,-1470.0],[85,777.82,-777.82,-1470.0],[86,284.7,-1062.52,-1470.0],[87,-284.7,-1062.52,-1470.0],
[88,-777.82,-777.82,-1470.0],[89,-1062.52,-284.7,-1470.0],[90,-1062.52,284.7,-1470.0],[91,-777.82,777.82,-1470.0],
[92,-284.7,1062.52,-1470.0],[93,284.7,1062.52,-1470.0],[94,777.82,777.82,-1470.0],[95,1062.52,284.7,-1470.0],
[96,2107.453,-35.3809,1475.0],[97,1084.3685,1807.4167,1475.0],[98,-1023.089,1842.7965,1475.0],[99,-2107.4532,35.372,1475.0],
[100,-1084.3637,-1807.4196,1475.0],[101,1023.0939,-1842.7938,1475.0],[102,-1566.1062,-1201.726,1606.0],[103,257.6573,-1957.1527,1606.0],
[104,1823.7752,-755.4323,1606.0],[105,1566.1118,1201.7187,1606.0],[106,-257.6717,1957.1508,1606.0],[107,-1823.7725,755.4389,1606.0],
[108,2107.453,-35.3809,-1475.0],[109,1084.3685,1807.4167,-1475.0],[110,-1023.089,1842.7965,-1475.0],[111,-2107.4532,35.372,-1475.0],
[112,-1084.3637,-1807.4196,-1475.0],[113,1023.0939,-1842.7938,-1475.0],[114,-1566.1062,-1201.726,-1606.0],[115,257.6573,-1957.1527,-1606.0],
[116,1823.7752,-755.4323,-1606.0],[117,1566.1118,1201.7187,-1606.0],[118,-257.6717,1957.1508,-1606.0],[119,-1823.7725,755.4389,-1606.0]])
            
            highlist = []
            lowlist = []
            
            vtx = []
            pos = []

            
            for j in range(nEvent):
                
                
                highlist.append(f['high'][j])
                lowlist.append(f['low'][j])
                vtx.append(f['real_vtx'][j])
                pos.append(pmt_pos[:96,1:4])
   
            


            
            self.highList.append(highlist)
            self.lowList.append(lowlist)
            self.vtxList.append(vtx)
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
