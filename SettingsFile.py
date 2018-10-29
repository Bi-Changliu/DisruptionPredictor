# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:31:31 2018

@author: Bi ChangLiu
"""

SettingsDict=dict()


DeviceNameDict=dict()
DeviceNameDict['hl2a']=['HL2A','HL-2A','2A']
SettingsDict['DeviceNameDict']=DeviceNameDict


InputParamDict=dict()
InputParamDict['SignalDict']=dict()
InputParamDict['SignalDict']['IP']=dict()
InputParamDict['SignalDict']['IP']['ChannelNum']=1
InputParamDict['SignalDict']['IP']['FreqDoub']=1
InputParamDict['SignalDict']['IP']['SubDir']='PlasmaPara'
InputParamDict['SignalDict']['Bt']=dict()
InputParamDict['SignalDict']['Bt']['ChannelNum']=1
InputParamDict['SignalDict']['Bt']['FreqDoub']=1
InputParamDict['SignalDict']['Bt']['SubDir']='Magnetic'
InputParamDict['SignalDict']['StoredEnergy']=dict()
InputParamDict['SignalDict']['StoredEnergy']['ChannelNum']=1
InputParamDict['SignalDict']['StoredEnergy']['FreqDoub']=10
InputParamDict['SignalDict']['StoredEnergy']['SubDir']='PlasmaPara'
InputParamDict['SignalDict']['Ne_R0']=dict()
InputParamDict['SignalDict']['Ne_R0']['ChannelNum']=1
InputParamDict['SignalDict']['Ne_R0']['FreqDoub']=100
InputParamDict['SignalDict']['Ne_R0']['SubDir']='PlasmaPara'
InputParamDict['0DSignalNum']=2
InputParamDict['Fs']=1000
InputParamDict['EmergencyDampRate']=float(30)
InputParamDict['MinDataLen']=150
SettingsDict['InputParamDict']=InputParamDict

ModelParamDict=dict()
ModelParamDict['CNNForStoredEnergy']=dict()
ModelParamDict['CNNForStoredEnergy']['Conv1KernelLength']=5
ModelParamDict['CNNForStoredEnergy']['Conv1KernelNum']=8
ModelParamDict['CNNForStoredEnergy']['Pooling1Length']=2
ModelParamDict['CNNForStoredEnergy']['Pooling1Stride']=1
ModelParamDict['CNNForStoredEnergy']['Conv2KernelLength']=5
ModelParamDict['CNNForStoredEnergy']['Conv2KernelNum']=8
ModelParamDict['CNNForStoredEnergy']['Pooling2Length']=2
ModelParamDict['CNNForStoredEnergy']['Pooling2Stride']=1
ModelParamDict['CNNForStoredEnergy']['FC1CellNum']=4
ModelParamDict['CNNForStoredEnergy']['FC2CellNum']=1
ModelParamDict['CNNForNe_R0']=dict()
ModelParamDict['CNNForNe_R0']['Conv1KernelLength']=30
ModelParamDict['CNNForNe_R0']['Conv1KernelNum']=16
ModelParamDict['CNNForNe_R0']['Pooling1Length']=4
ModelParamDict['CNNForNe_R0']['Pooling1Stride']=2
ModelParamDict['CNNForNe_R0']['Conv2KernelLength']=20
ModelParamDict['CNNForNe_R0']['Conv2KernelNum']=16
ModelParamDict['CNNForNe_R0']['Pooling2Length']=4
ModelParamDict['CNNForNe_R0']['Pooling2Stride']=2
ModelParamDict['CNNForNe_R0']['FC1CellNum']=10
ModelParamDict['CNNForNe_R0']['FC2CellNum']=3
ModelParamDict['RNNForAllSignals']=dict()
ModelParamDict['RNNForAllSignals']['LayerNum']=5
ModelParamDict['RNNForAllSignals']['HiddenSize']=6
ModelParamDict['RNNForAllSignals']['FC1CellNum']=6
ModelParamDict['RNNForAllSignals']['FC2CellNum']=1

SettingsDict['ModelParamDict']=ModelParamDict

TrainParamDict=dict()
TrainParamDict['Device']='HL2A'
TrainParamDict['DisrupRatio']=0.5
TrainParamDict['GpuNum']=2
TrainParamDict['BatchNum']=dict()
TrainParamDict['BatchNum']['BatchForOneTime']=20
TrainParamDict['BatchNum']['ValBatchSize']=100
TrainParamDict['OptiMizerType']='SGD'
TrainParamDict['SGD']=dict()
TrainParamDict['SGD']['EpochForLearningRate']=[5000,20000,5000]
TrainParamDict['SGD']['LearningRateList']=[0.03,0.01,0.003]
TrainParamDict['Adam']=dict()
TrainParamDict['Adam']['EpochNum']=[30000]
TrainParamDict['Adam']['LearningRateMax']=0.03
TrainParamDict['Adam']['LearningRateMin']=0.003
TrainParamDict['Adam']['DecaySpeed']=10000
TrainParamDict['StartCheckpoint']=''
TrainParamDict['SummariesDir']=r"C:/yzy/D盘/研究\DisruptionPredictor/2018102901/tensorboard"
TrainParamDict['TrainDir']=r"C:/yzy/D盘/研究\DisruptionPredictor/2018102901/checkpoint"
TrainParamDict['EvalStepInterval']=100
TrainParamDict['SaveStepInterval']=3000
SettingsDict['TrainParamDict']=TrainParamDict

TestParamDict=dict()
TestParamDict['Device']='HL2A'
TestParamDict['BatchNum']=dict()
TestParamDict['BatchNum']['BatchForOneTime']=20
TestParamDict['BatchNum']['TestBatchSize']=100
SettingsDict['TestParamDict']=TestParamDict