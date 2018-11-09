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
InputParamDict['HDF5FilePath']=r"\\192.168.9.242\hdf/"
InputParamDict['SignalDict']=dict()
InputParamDict['SignalDict']['IP']=dict()
InputParamDict['SignalDict']['IP']['ChannelNum']=1
InputParamDict['SignalDict']['IP']['FreqDoub']=1
InputParamDict['SignalDict']['IP']['SubDir']='PlasmaPara'
InputParamDict['SignalDict']['IP']['SubDir1']=None
InputParamDict['SignalDict']['Bt']=dict()
InputParamDict['SignalDict']['Bt']['ChannelNum']=1
InputParamDict['SignalDict']['Bt']['FreqDoub']=1
InputParamDict['SignalDict']['Bt']['SubDir']='Magnetic'
InputParamDict['SignalDict']['Bt']['SubDir1']=None
InputParamDict['SignalDict']['StoredEnergy']=dict()
InputParamDict['SignalDict']['StoredEnergy']['ChannelNum']=1
InputParamDict['SignalDict']['StoredEnergy']['FreqDoub']=10
InputParamDict['SignalDict']['StoredEnergy']['SubDir']='PlasmaPara'
InputParamDict['SignalDict']['StoredEnergy']['SubDir1']=None
InputParamDict['SignalDict']['Ne_R0']=dict()
InputParamDict['SignalDict']['Ne_R0']['ChannelNum']=1
InputParamDict['SignalDict']['Ne_R0']['FreqDoub']=100
InputParamDict['SignalDict']['Ne_R0']['SubDir']='PlasmaPara'
InputParamDict['SignalDict']['Ne_R0']['SubDir1']=None
InputParamDict['SignalDict']['P_Rad']=dict()
InputParamDict['SignalDict']['P_Rad']['ChannelNum']=1
InputParamDict['SignalDict']['P_Rad']['FreqDoub']=1
InputParamDict['SignalDict']['P_Rad']['SubDir']='PlasmaPara'
InputParamDict['SignalDict']['P_Rad']['SubDir1']=None
InputParamDict['SignalDict']['Mir_Pol_A']=dict()
InputParamDict['SignalDict']['Mir_Pol_A']['ChannelNum']=1
InputParamDict['SignalDict']['Mir_Pol_A']['FreqDoub']=100
InputParamDict['SignalDict']['Mir_Pol_A']['SubDir']='Diagnostic'
InputParamDict['SignalDict']['Mir_Pol_A']['SubDir1']='Mirnov'
InputParamDict['0DSignalNum']=3
InputParamDict['Fs']=1000
InputParamDict['EmergencyDampRate']=float(50)
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
ModelParamDict['CNNForMir_Pol_A']=dict()
ModelParamDict['CNNForMir_Pol_A']['Conv1KernelLength']=30
ModelParamDict['CNNForMir_Pol_A']['Conv1KernelNum']=16
ModelParamDict['CNNForMir_Pol_A']['Pooling1Length']=4
ModelParamDict['CNNForMir_Pol_A']['Pooling1Stride']=2
ModelParamDict['CNNForMir_Pol_A']['Conv2KernelLength']=20
ModelParamDict['CNNForMir_Pol_A']['Conv2KernelNum']=16
ModelParamDict['CNNForMir_Pol_A']['Pooling2Length']=4
ModelParamDict['CNNForMir_Pol_A']['Pooling2Stride']=2
ModelParamDict['CNNForMir_Pol_A']['FC1CellNum']=10
ModelParamDict['CNNForMir_Pol_A']['FC2CellNum']=3
ModelParamDict['RNNForAllSignals']=dict()
ModelParamDict['RNNForAllSignals']['LayerNum']=2
ModelParamDict['RNNForAllSignals']['HiddenSize']=\
    ModelParamDict['CNNForMir_Pol_A']['FC2CellNum']+\
    ModelParamDict['CNNForNe_R0']['FC2CellNum']+\
    ModelParamDict['CNNForStoredEnergy']['FC2CellNum']+\
    InputParamDict['0DSignalNum']
ModelParamDict['RNNForAllSignals']['FC1CellNum']=6
ModelParamDict['RNNForAllSignals']['FC2CellNum']=1

SettingsDict['ModelParamDict']=ModelParamDict

TrainParamDict=dict()
TrainParamDict['Device']='HL2A'
TrainParamDict['DisrupRatio']=0.9
TrainParamDict['StartShotNum']=20000
TrainParamDict['StopShotNum']=33000
TrainParamDict['ValRatio']=0.1
TrainParamDict['TfRecordsDir']=r"D:/yzy/DisruptionPredictor/TfRecords"
TrainParamDict['GpuNum']=2
TrainParamDict['BatchNum']=dict()
TrainParamDict['BatchNum']['BatchForOneTime']=500
TrainParamDict['BatchNum']['ValBatchSize']=2000
TrainParamDict['OptiMizerType']='SGD'
TrainParamDict['SGD']=dict()
TrainParamDict['SGD']['EpochForLearningRate']=[5000,20000,5000]
TrainParamDict['SGD']['LearningRateList']=[0.3,0.03,0.003]
TrainParamDict['Adam']=dict()
TrainParamDict['Adam']['EpochNum']=[30000]
TrainParamDict['Adam']['LearningRateMax']=0.03
TrainParamDict['Adam']['LearningRateMin']=0.003
TrainParamDict['Adam']['DecaySpeed']=10000
TrainParamDict['StartCheckpoint']=''
TrainParamDict['SummariesDir']=r"D:/yzy/DisruptionPredictor/2018110801/tensorboard"
TrainParamDict['TrainDir']=r"D:/yzy/DisruptionPredictor/2018110801/checkpoint"
TrainParamDict['EvalStepInterval']=100
TrainParamDict['SaveStepInterval']=3000
SettingsDict['TrainParamDict']=TrainParamDict

TestParamDict=dict()
TestParamDict['Device']='HL2A'
TestParamDict['StartShotNum']=33000
TestParamDict['StopShotNum']=35000
TestParamDict['BatchNum']=dict()
TestParamDict['BatchNum']['BatchForOneTime']=20
TestParamDict['BatchNum']['TestBatchSize']=10000
SettingsDict['TestParamDict']=TestParamDict