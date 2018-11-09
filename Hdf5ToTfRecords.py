# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:21:46 2018
This file aims to write all the hdf5 data into a tfrecord file, so that they 
can be read parallized

@author: Bi ChangLiu
"""

import h5py
import numpy as np
import random
import tensorflow as tf

from CShotNumList import CShotNumList
import MysqlDatabaseUtils

from SettingsFile import SettingsDict

def GetHdf5Data(ShotNum,Device,DisrupShotsMySqlProccessor):
    SignalDict=SettingsDict['InputParamDict']['SignalDict']
    EmergencyDampRate=SettingsDict['InputParamDict']['EmergencyDampRate']
    Fs=SettingsDict['InputParamDict']['Fs']
    HDF5FilePath=SettingsDict['InputParamDict']['HDF5FilePath']
    Device = Device.upper()
    DirNumber = ShotNum // 200 * 200
    if Device in ["HL-2A","HL2A","2A"]:
        FilePathName = HDF5FilePath + '2A/' + str(DirNumber)
        FilePathName = FilePathName + '/HL-2A=' + str(ShotNum)+'=PhysicsDB.h5'
        Device="hl2a"
    elif Device in ["J-TEXT","JTEXT","TEXT"]:
        FilePathName = HDF5FilePath + "EAST/" + str(DirNumber)
        FilePathName = FilePathName + "/EAST=" + str(ShotNum)+"=PhysicsDB.h5"
        Device="jtext"
    elif Device in ["EAST"]:
        FilePathName = HDF5FilePath + "J-TEXT/" + str(DirNumber)
        FilePathName = FilePathName + "/J-TEXT=" + str(ShotNum)+"=PhysicsDB.h5"
        Device="east"
    DataPipe = h5py.File(FilePathName,"r")

    DisrupParam=DisrupShotsMySqlProccessor.ReadByShotNum(ShotNum,Device)
    TStart=DisrupParam['TFinishRisingStage']/1000.0
    if DisrupParam['bDisruptive']:
        TStop=DisrupParam['TDisruptionStart']/1000.0
    else:
        TStop=(DisrupParam['TFinishRisingStage']+DisrupParam['TShotLength'])/1000.0
    
    DataList=list(range(int(Fs*TStop)-int(Fs*TStart)))
    for Idx in range(len(DataList)):
        DataList[Idx]=dict()
        DataList[Idx]['0DSignals']=list()
    for SignalName in SignalDict.keys():
        SubDir = SignalDict[SignalName]['SubDir']
        SubDir1 = SignalDict[SignalName]['SubDir1']
        if SubDir1 is None:
            Signal = DataPipe[SubDir][SignalName]
        else:
            Signal = DataPipe[SubDir][SubDir1][SignalName]
        for (Idx,TimeIdx) in enumerate(range(int(Fs*TStart),int(Fs*TStop))):
            if SignalDict[SignalName]['ChannelNum']==1:
                if SignalDict[SignalName]['FreqDoub']==1:
                    DataList[Idx]['0DSignals'].append(Signal[TimeIdx])
                else:
                    DataList[Idx][SignalName]=np.array(Signal[TimeIdx])
            else:
                IdxStart=TimeIdx*SignalDict[SignalName]['FreqDoub']
                IdxStop=(TimeIdx+1)*SignalDict[SignalName]['FreqDoub']
                DataList[SignalName]=np.array(Signal[IdxStart:IdxStop])
    for PointData in DataList:
        PointData['0DSignals']=np.array(PointData['0DSignals'])
    if DisrupParam['bDisruptive']:
        TimeToDisruption=np.array(range(int(Fs*TStart)-int(Fs*TStop),0))
        EmergencyValue=np.floor(np.array(100*np.exp(TimeToDisruption/EmergencyDampRate)))
    else:
        EmergencyValue=-np.ones(len(DataList))
    return DataList,EmergencyValue

def DataListToDataSliceDict(DataList,EmergencyValue):
    SignalDict=SettingsDict['InputParamDict']['SignalDict']
    MinDataLen=len(DataList)
    DataSliceDict=dict()
    for SignalName in SignalDict.keys():
        ChannelNum=SignalDict[SignalName]['ChannelNum']
        FreqDoub=SignalDict[SignalName]['FreqDoub']
        if FreqDoub==1 or ChannelNum==1:
            if FreqDoub==1 and ChannelNum==1:
                continue
            DataSliceDict[SignalName]=np.zeros([MinDataLen,ChannelNum*FreqDoub])
            for TimeIdx in range(MinDataLen):
                DataSliceDict[SignalName][TimeIdx,:]=DataList[TimeIdx][SignalName]
        else:
            DataSliceDict[SignalName]=np.zeros([MinDataLen,ChannelNum,FreqDoub])
            for TimeIdx in range(MinDataLen):
                DataSliceDict[SignalName][TimeIdx,:,:]=DataList[TimeIdx][SignalName]
    SignalName='0DSignals'
    FreqDoub=1
    ChannelNum=DataList[0]['0DSignals'].shape[0]
    DataSliceDict[SignalName]=np.zeros([MinDataLen,ChannelNum*FreqDoub])
    for TimeIdx in range(MinDataLen):
        DataSliceDict[SignalName][TimeIdx,:]=DataList[TimeIdx][SignalName]
    SignalName='EmergencyValue'
    FreqDoub=1
    ChannelNum=1
    DataSliceDict[SignalName]=np.zeros([MinDataLen,ChannelNum*FreqDoub])
    for TimeIdx in range(MinDataLen):
        DataSliceDict[SignalName][TimeIdx,:]=EmergencyValue[TimeIdx]
    return DataSliceDict
    
TrainDevice=SettingsDict['TrainParamDict']['Device']
TrainDisrupRatio=SettingsDict['TrainParamDict']['DisrupRatio']
TrainStartShotNum=SettingsDict['TrainParamDict']['StartShotNum']
TrainStopShotNum=SettingsDict['TrainParamDict']['StopShotNum']
ValRatio=SettingsDict['TrainParamDict']['ValRatio']
TfRecordsDir=SettingsDict['TrainParamDict']['TfRecordsDir']
TestDevice=SettingsDict['TestParamDict']['Device']
TestStartShotNum=SettingsDict['TestParamDict']['StartShotNum']
TestStopShotNum=SettingsDict['TestParamDict']['StopShotNum']

MinDataLen=SettingsDict['InputParamDict']['MinDataLen']

TrainWriter = tf.python_io.TFRecordWriter(TfRecordsDir+'/TrainData.tfrecords')
ValWriter = tf.python_io.TFRecordWriter(TfRecordsDir+'/ValData.tfrecords')
TestWriter = tf.python_io.TFRecordWriter(TfRecordsDir+'/TestData.tfrecords')

DisrupShotsMySqlProccessor=MysqlDatabaseUtils.CDisrupShotsMySqlProccessor()
TrainShotNumList=CShotNumList(TrainDevice)
TestShotNumList=CShotNumList(TestDevice)

TrainShotNumbers=TrainShotNumList.GetShotNumByRangeAndRatio(TrainStartShotNum,
                                                            TrainStopShotNum,
                                                            TrainDisrupRatio)
TestShotNumbers=TrainShotNumList.GetShotNumByRangeAndRatio(TestStartShotNum,
                                                           TestStopShotNum)

DataLists=list()
EmergencyValues=list()
MaxLen=-np.Inf
for ShotNum in TrainShotNumbers:
    DataList,EmergencyValue=GetHdf5Data(int(ShotNum),TrainDevice,DisrupShotsMySqlProccessor)
    DataLists.append(DataList)
    EmergencyValues.append(EmergencyValue)
    if len(DataList)>MaxLen:
        MaxLen=len(DataList)

TimeIdxList=list(range(int((MaxLen-MinDataLen)/10)+1))
np.random.shuffle(TimeIdxList)
for TimeIdx in TimeIdxList:
    for (ShotIdx,DataList) in enumerate(DataLists):
        EmergencyValue=EmergencyValues[ShotIdx]
        StartIdx=TimeIdx*10
        if StartIdx>len(DataList)-MinDataLen+10:
            continue
        else:
            StartIdx=min([len(DataList)-MinDataLen,StartIdx])
            StopIdx=StartIdx+MinDataLen
            DataSliceDict=DataListToDataSliceDict(DataList[StartIdx:StopIdx],
                                                  EmergencyValue[StartIdx:StopIdx])
            Features={}
            for SignalName in DataSliceDict.keys():
                Features[SignalName]=tf.train.Feature(bytes_list=tf.train.BytesList\
                        (value=[DataSliceDict[SignalName].tobytes()]))
            TfFeatures=tf.train.Features(feature=Features)
            TfExample=tf.train.Example(features=TfFeatures)
            TfSerialized=TfExample.SerializeToString()
                
            SetChooser=random.random()
            if (SetChooser<ValRatio):
                ValWriter.write(TfSerialized)
            else:
                TrainWriter.write(TfSerialized)
            
DataLists=list()
EmergencyValues=list()
MaxLen=-np.Inf
for ShotNum in TestShotNumbers:
    DataList,EmergencyValue=GetHdf5Data(int(ShotNum),TestDevice,DisrupShotsMySqlProccessor)
    DataLists.append(DataList)
    EmergencyValues.append(EmergencyValue)
    if len(DataList)>MaxLen:
        MaxLen=len(DataList)

TimeIdxList=list(range(int((MaxLen-MinDataLen)/10)+1))
np.random.shuffle(TimeIdxList)
for TimeIdx in TimeIdxList:
    for (ShotIdx,DataList) in enumerate(DataLists):
        EmergencyValue=EmergencyValues[ShotIdx]
        StartIdx=TimeIdx*10
        if StartIdx>len(DataList)-MinDataLen+10:
            continue
        else:
            StartIdx=min([len(DataList)-MinDataLen,StartIdx])
            StopIdx=StartIdx+10
            DataSliceDict=DataListToDataSliceDict(DataList[StartIdx:StopIdx],
                                                  EmergencyValue[StartIdx:StopIdx])
            Features={}
            for SignalName in DataSliceDict.keys():
                Features[SignalName]=tf.train.Feature(bytes_list=tf.train.BytesList\
                        (value=[DataSliceDict[SignalName].tobytes()]))
            TfFeatures=tf.train.Features(feature=Features)
            TfExample=tf.train.Example(features=TfFeatures)
            TfSerialized=TfExample.SerializeToString()
            TestWriter.write(TfSerialized)

TrainWriter = TrainWriter.close()
ValWriter = ValWriter.close()
TestWriter = TestWriter.close()