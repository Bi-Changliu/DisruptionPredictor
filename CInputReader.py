# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 16:26:21 2018
This file defines the CInputReader class. The class get ShotNum from CShotNumList,
read data from hdf5 file in the dataserver and reshepe them into a 1D-list,
which contains all the signal dictionary corresponding to each time point. Each 
dictionary will contains a 1D-array which consists of all the space-independent
signals and some 1D or 2D-arrays which correspongding to each profile signals

@author: Bi ChangLiu
"""

import h5py
import numpy as np

import MysqlDatabaseUtils
from CShotNumList import CShotNumList
from SettingsFile import SettingsDict


class CInputReader(object):
    def __init__(self,DeviceList):
        self.SignalDict=SettingsDict['InputParamDict']['SignalDict']
        self.EmergencyDampRate=SettingsDict['InputParamDict']['EmergencyDampRate']
        self.MinDataLen=SettingsDict['InputParamDict']['MinDataLen']
        self.Fs=SettingsDict['InputParamDict']['Fs']
        self.HDF5FilePath=r"\\192.168.9.242\hdf/"
        self.DisrupShotsMySqlProccessor=MysqlDatabaseUtils.CDisrupShotsMySqlProccessor()
        self.ShotNumLists=dict()
        for Device in DeviceList:
            self.ShotNumLists[Device]=CShotNumList(Device)
        
    def ReadData(self,ShotNum,Device):
        Device = Device.upper()
        DirNumber = ShotNum // 200 * 200
        if Device in ["HL-2A","HL2A","2A"]:
            FilePathName = self.HDF5FilePath + '2A/' + str(DirNumber)
            FilePathName = FilePathName + '/HL-2A=' + str(ShotNum)+'=PhysicsDB.h5'
            Device="hl2a"
        elif Device in ["J-TEXT","JTEXT","TEXT"]:
            FilePathName = self.HDF5FilePath + "EAST/" + str(DirNumber)
            FilePathName = FilePathName + "/EAST=" + str(ShotNum)+"=PhysicsDB.h5"
            Device="jtext"
        elif Device in ["EAST"]:
            FilePathName = self.HDF5FilePath + "J-TEXT/" + str(DirNumber)
            FilePathName = FilePathName + "/J-TEXT=" + str(ShotNum)+"=PhysicsDB.h5"
            Device="east"
        DataPipe = h5py.File(FilePathName,"r")
        
        DisrupParam=self.DisrupShotsMySqlProccessor.ReadByShotNum(ShotNum,Device)
        TStart=DisrupParam['TFinishRisingStage']/1000.0
        if DisrupParam['bDisruptive']:
            TStop=DisrupParam['TDisruptionStart']/1000.0
        else:
            TStop=(DisrupParam['TFinishRisingStage']+DisrupParam['TShotLength'])/1000.0
        
        DataList=list(range(int(self.Fs*TStop)-int(self.Fs*TStart)))
        for Idx in range(len(DataList)):
            DataList[Idx]=dict()
            DataList[Idx]['0DSignals']=list()
        for SignalName in self.SignalDict.keys():
            SubDir = self.SignalDict[SignalName]['SubDir']
            Signal = DataPipe[SubDir][SignalName]
            for (Idx,TimeIdx) in enumerate(range(int(self.Fs*TStart),int(self.Fs*TStop))):
                if self.SignalDict[SignalName]['ChannelNum']==1:
                    if self.SignalDict[SignalName]['FreqDoub']==1:
                        DataList[Idx]['0DSignals'].append(Signal[TimeIdx])
                    else:
                        DataList[Idx][SignalName]=np.array(Signal[TimeIdx])
                else:
                    IdxStart=TimeIdx*self.SignalDict[SignalName]['FreqDoub']
                    IdxStop=(TimeIdx+1)*self.SignalDict[SignalName]['FreqDoub']
                    DataList[SignalName]=np.array(Signal[IdxStart:IdxStop])
        for PointData in DataList:
            PointData['0DSignals']=np.array(PointData['0DSignals'])
        if DisrupParam['bDisruptive']:
            TimeToDisruption=np.array(range(int(self.Fs*TStart)-int(self.Fs*TStop),0))
            EmergencyValue=np.floor(np.array(100*np.exp(TimeToDisruption/self.EmergencyDampRate)))
        else:
            EmergencyValue=np.zeros(len(DataList))
        return DataList,EmergencyValue
    
    def ReadBatchData(self,BatchNum,DisrupRatio,Device):
        ShotNums=self.ShotNumLists[Device].GetShotNumBatch(BatchNum,DisrupRatio)
        DataLists=list()
        EmergencyLists=list()
        for ShotNum in ShotNums:
            print(ShotNum)
            DataList,EmergencyList=self.ReadData(int(ShotNum),Device)
            IdxStart=np.random.randint(0,len(DataList)-self.MinDataLen)
            DataLists.append(DataList[IdxStart:IdxStart+self.MinDataLen])
            EmergencyLists.append(EmergencyList[IdxStart:IdxStart+self.MinDataLen])
        EmergencyValue=np.zeros([BatchNum,self.MinDataLen])
        for ShotNumIdx in range(len(EmergencyLists)):
            EmergencyValue[ShotNumIdx,:]=EmergencyLists[ShotNumIdx]
        BatchData=dict()
        for SignalName in self.SignalDict.keys():
            ChannelNum=self.SignalDict[SignalName]['ChannelNum']
            FreqDoub=self.SignalDict[SignalName]['FreqDoub']
            if FreqDoub==1 or ChannelNum==1:
                if FreqDoub==1 and ChannelNum==1:
                    continue
                BatchData[SignalName]=np.zeros([BatchNum,self.MinDataLen,ChannelNum*FreqDoub])
                for ShotNumIdx in range(BatchNum):
                    for TimeIdx in range(self.MinDataLen):
                        BatchData[SignalName][ShotNumIdx,TimeIdx,:]=\
                            DataLists[ShotNumIdx][TimeIdx][SignalName]
            else:
                BatchData[SignalName]=np.zeros([BatchNum,self.MinDataLen,ChannelNum,FreqDoub])
                for ShotNumIdx in range(BatchNum):
                    for TimeIdx in range(self.MinDataLen):
                        BatchData[SignalName][ShotNumIdx,TimeIdx,:,:]=\
                            DataLists[ShotNumIdx][TimeIdx][SignalName]
        SignalName='0DSignals'
        FreqDoub=1
        ChannelNum=DataLists[0][0]['0DSignals'].shape[0]
        BatchData[SignalName]=np.zeros([BatchNum,self.MinDataLen,ChannelNum*FreqDoub])
        for ShotNumIdx in range(BatchNum):
            for TimeIdx in range(self.MinDataLen):
                BatchData[SignalName][ShotNumIdx,TimeIdx,:]=\
                    DataLists[ShotNumIdx][TimeIdx][SignalName]
        return BatchData,EmergencyValue