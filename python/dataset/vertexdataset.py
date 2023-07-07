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
        
        pmts_pos = torch.Tensor(self.pmts_posList[fileIdx][idx])
        features = torch.Tensor(self.featureList[fileIdx][idx])
        total_charges = torch.Tensor(self.totalcharge[fileIdx][idx])
        true_vertexs = torch.Tensor(self.true_vtxList[fileIdx][idx])
        true_energys = torch.Tensor(self.true_energyList[fileIdx][idx])
        jade_vertexs = torch.Tensor(self.jade_vtxList[fileIdx][idx])
        jade_energys = torch.Tensor(self.jade_energyList[fileIdx][idx])
#         jade_hwf = torch.Tensor(self.high_wfList[fileIdx][idx])
#         jade_lwf = torch.Tensor(self.low_wfList[fileIdx][idx])
        
                
        data = PyGData(x = features, pos = pmts_pos, y = true_vertexs)
        data.javtx = jade_vertexs
        data.jae = jade_energys
        data.tre = true_energys
        data.tq = total_charges
#         data.hwf = jade_hwf
#         data.lwf = jade_lwf

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
    def initialize(self, geo, itype, tev, output):
        if self.isLoaded: return

        print(self.sampleInfo)
        procNames = list(self.sampleInfo['procName'].unique())

        self.procList = []
        self.pmts_posList = []
        self.featureList = []
        self.totalcharge = []
        self.true_vtxList = []
        self.true_energyList = []
        self.jade_vtxList = []
        self.jade_energyList = []
        self.high_wfList = []
        self.low_wfList = []
                
        #### file num check
        nFiles = len(self.sampleInfo)
        #### Load event contents
        for i, fName in enumerate(self.sampleInfo['fileName']):
            
            #### file load and check event num
            f = h5py.File(fName,'r', libver='latest', swmr=True)['events']
            nEvent = len(f['pmtQ'])
            self.sampleInfo.loc[i, 'nEvent'] = nEvent

            #### detector geometry information
            if geo == 1:
                pos_file = 'python/detector_geometry/jsns_geometry_pos.csv' ### 96 PMTs
            elif geo == 2:
                pos_file = 'python/detector_geometry/jsns_geometry_pos2.csv' ### 120 PMTs
                
            ff = pd.read_csv(pos_file,header=0)

            #### empty list 
            pmts_pos = []
            feature = []
            total_ch = []
            true_vtx = []
            true_energy = []
            jade_vtx = []
            jade_energy = []
#             high_wf = []
#             low_wf = []
            #### each file event append list
            for j in range(nEvent):
                
                #### ftype option remove
                #### only use h5 file
                #### only select data type
                #### h5 file high waveform
                if itype == 1:
                    
                    feature.append(f['high'][j])
                    true_vtx.append(f['vtx'][j])
                    pmts_pos.append(np.array(ff))
                    
                #### h5 file low waveform
                elif itype == 2:
                    
                    feature.append(f['low'][j])
                    true_vtx.append(f['vtx'][j])
                    pmts_pos.append(np.array(ff))
                    
                #### h5 file sum waveform
                elif itype == 3:
                    
                    feature.append(f['low'][j]+f['high'][j])
                    true_vtx.append(f['vtx'][j])
                    pmts_pos.append(np.array(ff))
                    
                #### h5 file charge
                elif itype == 0:
                    
                    feature.append(f['jade_pmtQ'][j][0:96].reshape(-1,1))
                    total_ch.append(np.ones([96,1])*np.array(f['jade_pmtQ'][j][0:96].sum()))
                    true_vtx.append(f['vertex'][j]/1000)
                    pmts_pos.append(np.array(ff))
                    true_energy.append(f['E'][j])
                    jade_vtx.append(f['jade_vertex'][j])
                    jade_energy.append(f['jade_E'][j])
#                     high_wf.append(f['high_wf'][j])
#                     low_wf.append(f['low_wf'][j])
                                    

            #### all event append list
            self.pmts_posList.append(pmts_pos)
            self.featureList.append(feature)
            self.totalcharge.append(total_ch)
            self.true_vtxList.append(true_vtx)
            self.true_energyList.append(true_energy)
            self.jade_vtxList.append(jade_vtx)
            self.jade_energyList.append(jade_energy)
#             self.high_wfList.append(high_wf)
#             self.low_wfList.append(low_wf)
            
        #### save sampleInfo file in train result path
        SI = self.sampleInfo
        if tev == 1:
            SI.to_csv('result/'+output + '/training_sampleInfo.csv')
        else:
            SI.to_csv('result/'+output + '/evaluation_sampleInfo.csv')
        
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
