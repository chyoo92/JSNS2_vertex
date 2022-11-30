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



   
        
class vertexdataset(PyGDataset):
    def __init__(self, **kwargs):
        super(vertexdataset, self).__init__(None, transform=None, pre_transform=None)
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
        feature = torch.Tensor(self.featureList[fileIdx][idx])
        jvertex = torch.Tensor(self.jvtxList[fileIdx][idx])
        energys = torch.Tensor(self.energyList[fileIdx][idx])
        
        data = PyGData(x = feature, pos = pos, y = vertex)
        data.jvtx = jvertex
        data.energy = energys
   

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
    def initialize(self, geo, itype, ftype):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())


        self.featureList = []
 
        self.vtxList = []
        self.posList = []
        self.procList = []
        self.jvtxList = []
        self.energyList = []
        
        
        nFiles = len(self.sampleInfo)
        ## Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):

            if ftype == 0:
                f = pd.read_csv(fName)

                nEvent = len(np.array(f))
            elif ftype == 1:
                f = h5py.File(fName,'r', libver='latest', swmr=True)['events']
                keys = f.keys()
                nEvent = len(str(keys)[15:-2].split(',')[-2])
  
            self.sampleInfo.loc[i, 'nEvent'] = nEvent
            
            if geo == 1:
                pos_file = 'python/detector_geometry/jsns_geometry_pos.csv'
            elif geo == 2:
                pos_file = 'python/detector_geometry/sphere_geometry_pos.csv'
            elif geo == 3:
                pos_file = 'python/detector_geometry/cylinder_geometry_pos.csv'
            elif geo == 0:
                pos_file = 'python/detector_geometry/jsns_geometry_pos2.csv'
                
            ff = pd.read_csv(pos_file,header=0)
            
            featurelist = []
            
            vtx = []
            pos = []
            j_vtx = []
            energy = []
            
            
            for j in range(nEvent):
                
                #### h5 file
                if ftype == 1:
                    
                    #### h5 file high waveform
                    if itype == 1:
                        
                        featurelist.append(f['high'][j])
                        
                        vtx.append(f['vtx'][j])
                        
                        pos.append(np.array(ff))
                        
                    #### h5 file low waveform
                    elif itype == 2:
                        
                        featurelist.append(f['low'][j])
                        
                        vtx.append(f['vtx'][j])
                        
                        pos.append(np.array(ff))
                        
                    #### h5 file sum waveform
                    elif itype == 3:
                        
                        featurelist.append(f['low'][j]+f['high'][j])
                        
                        vtx.append(f['vtx'][j])
                        
                        pos.append(np.array(ff))
                        
                    #### h5 file charge
                    elif itype == 0:
                        
                        featurelist.append(f['pmtQ'][j][0:96].reshape(-1,1))

                        vtx.append(f['vertex'][j])

                        pos.append(np.array(ff))
                        if len(f['jade_vertex'][j]) > 0:
                            j_vtx.append(f['jade_vertex'][j])
                        else:
                            j_vtx.append([0,0,0])
                        
                        energy.append([int(fName.split('_')[-2][:-3])])
                        
               
                #### csv file 
                elif ftype == 0:
                    #### csv file charge
                    if itype == 0:
                        pos.append(np.array(ff))
                
   
            


            if ftype == 1:
                self.featureList.append(featurelist)
                self.vtxList.append(vtx)
                self.posList.append(pos)
                self.jvtxList.append(j_vtx)
                self.energyList.append(energy)
            
            elif ftype == 0:
                self.featureList.append(np.array(f)[:,17:17+len(ff)].reshape(-1,len(ff),1).tolist())
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
