# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:48:47 2018
This file controls the training

@author: Bi ChangLiu
"""
import os
import numpy as np
import tensorflow as tf
import CInputReader
import ModelBuilder
from SettingsFile import SettingsDict

#InputConf
BatchForOneTime=SettingsDict['TrainParamDict']['BatchNum']['BatchForOneTime']
ValBatchSize=SettingsDict['TrainParamDict']['BatchNum']['ValBatchSize']
TestBatchSize=SettingsDict['TestParamDict']['BatchNum']['TestBatchSize']
TrainBatchSize=BatchForOneTime

#OptimizerConf
OptiMizerType=SettingsDict['TrainParamDict']['OptiMizerType'] #Adam or SGD
#if SGD
EpochForLearningRate=SettingsDict['TrainParamDict']['SGD']['EpochForLearningRate']
LearningRateList=SettingsDict['TrainParamDict']['SGD']['LearningRateList']
#if Adam
EpochNum=SettingsDict['TrainParamDict']['Adam']['EpochNum']
LearningRateMax=SettingsDict['TrainParamDict']['Adam']['LearningRateMax']
LearningRateMin=SettingsDict['TrainParamDict']['Adam']['LearningRateMin']
DecaySpeed=SettingsDict['TrainParamDict']['Adam']['DecaySpeed']

StartCheckpoint=SettingsDict['TrainParamDict']['StartCheckpoint']
SummariesDir=SettingsDict['TrainParamDict']['SummariesDir']
TrainDir=SettingsDict['TrainParamDict']['TrainDir']
EvalStepInterval=SettingsDict['TrainParamDict']['EvalStepInterval']
SaveStepInterval=SettingsDict['TrainParamDict']['SaveStepInterval']
MinDataLen=SettingsDict['InputParamDict']['MinDataLen']

TrainDevice=SettingsDict['TrainParamDict']['Device']
TestDevice=SettingsDict['TestParamDict']['Device']
DeviceList=[TrainDevice,TestDevice]
DeviceList=list(set(DeviceList))
InputReader=CInputReader.CInputReader(DeviceList)

GpuNum=SettingsDict['TrainParamDict']['GpuNum']
DisrupRatio=SettingsDict['TrainParamDict']['DisrupRatio']

tf.reset_default_graph()
with tf.Session() as Sess:
    All0DSignals=tf.placeholder(tf.float32, 
                                [BatchForOneTime, MinDataLen, SettingsDict['InputParamDict']['0DSignalNum']], 
                                name='All0DSignals')
    StoredEnergy=tf.placeholder(tf.float32, 
                                [BatchForOneTime, MinDataLen, SettingsDict['InputParamDict']['SignalDict']['StoredEnergy']['FreqDoub']], 
                                name='All0DSignals')
    Ne_R0=tf.placeholder(tf.float32, 
                                [BatchForOneTime, MinDataLen, SettingsDict['InputParamDict']['SignalDict']['Ne_R0']['FreqDoub']], 
                                name='All0DSignals')
    EmergencyValue=tf.placeholder(tf.int64, 
                                  [BatchForOneTime, MinDataLen], 
                                  name='EmergencyValue')

    StoredEnergyCNNOutput,StoredEnergyDropoutProb = ModelBuilder.CNNForStoredEnergy(StoredEnergy, IsTraining=True)
    Ne_R0CNNOutput,Ne_R0DropoutProb = ModelBuilder.CNNForNe_R0(Ne_R0, IsTraining=True)
    
    AllSignals = tf.concat([All0DSignals,StoredEnergyCNNOutput,Ne_R0CNNOutput],2,name='AllSignals')
    
    RNNOutput,RNNDropoutProb=ModelBuilder.RNNForAllSignals(AllSignals)
    
    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('MeanSquaredError'):
        MeanSquaredError = tf.losses.mean_squared_error(EmergencyValue,RNNOutput)
    tf.summary.scalar('MeanSquaredError', MeanSquaredError)
    with tf.name_scope('train'), tf.control_dependencies([]):
        LearningRate = tf.placeholder(
            tf.float32, [], name='learning_rate_input')
        if OptiMizerType=='SGD':
            TrainStep=tf.train.GradientDescentOptimizer(LearningRate).minimize(MeanSquaredError)
        else:
            TrainStep=tf.train.AdamOptimizer(LearningRate).minimize(MeanSquaredError) 
    GlobalStep=tf.Variable(0,name='GlobalStep')
    IncrementGlobalStep=tf.assign(GlobalStep, GlobalStep+1)
    Saver=tf.train.Saver(tf.global_variables())
    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    MergedSummaries = tf.summary.merge_all()
    TrainWriter=tf.summary.FileWriter(SummariesDir+'/train')
    ValidationWriter=tf.summary.FileWriter(SummariesDir+'/validation')
    
    tf.global_variables_initializer().run()
    StartStep=1
    if StartCheckpoint:
        ModelBuilder.load_variables_from_checkpoint(Sess, StartCheckpoint)
        StartStep = GlobalStep.eval(session=Sess)

    #记载日志信息
    tf.logging.info('Training from step: %d ', StartStep)
    
    # Save graph.pbtxt.
    tf.train.write_graph(Sess.graph_def, TrainDir,'conv.pbtxt')
    
    if OptiMizerType=='SGD':
        EpochNum=sum(EpochForLearningRate)
    
    for EpochIndex in range(StartStep,EpochNum+1):
        if OptiMizerType=='SGD':
            for i in range(len(LearningRateList)):
                if sum(EpochForLearningRate[0:i])>EpochIndex:
                    break
            LearningRateInput = LearningRateList[i-1]
        else:
            LearningRateInput = LearningRateMin+(LearningRateMax-LearningRateMin)*np.exp(-EpochIndex/DecaySpeed)
            
        Data,EmergencyValueInput=InputReader.ReadBatchData(BatchForOneTime,DisrupRatio,TrainDevice)

        # Pull the audio samples we'll use for training.
        TrainSummary, TrainError, _, _ = Sess.run(
            [
                MergedSummaries, MeanSquaredError, TrainStep,
                IncrementGlobalStep
            ],
            feed_dict={
                All0DSignals            : Data['0DSignals'],
                StoredEnergy            : Data['StoredEnergy'],
                Ne_R0                   : Data['Ne_R0'],
                EmergencyValue          : EmergencyValueInput,
                LearningRate            : LearningRateInput,
                StoredEnergyDropoutProb : 0.5,
                Ne_R0DropoutProb        : 0.5,
                RNNDropoutProb          : 0.5
            })
        TrainWriter.add_summary(TrainSummary, EpochIndex)
        tf.logging.info('Step #%d: rate %f, error %.1f' %
                    (EpochIndex, LearningRateInput, TrainError))
        IsLastEpoch=(EpochIndex==EpochNum)
        
        #经过一定的training step就validate一次
        if (EpochIndex % EvalStepInterval)==0 or IsLastEpoch:
            AvgError = 0
            for i in range(0, ValBatchSize, BatchForOneTime):
                Data,EmergencyValueInput=InputReader.ReadBatchData(BatchForOneTime,DisrupRatio,TrainDevice)
                ValSummary, ValError = Sess.run(
                    [MergedSummaries, MeanSquaredError],
                    feed_dict={
                        All0DSignals            : Data['0DSignals'],
                        StoredEnergy            : Data['StoredEnergy'],
                        Ne_R0                   : Data['Ne_R0'],
                        EmergencyValue          : EmergencyValueInput,
                        StoredEnergyDropoutProb : 1.0,
                        Ne_R0DropoutProb        : 1.0,
                        RNNDropoutProb          : 1.0
                    })
                AvgError += (ValError*BatchForOneTime)/TestBatchSize
            ValidationWriter.add_summary(ValSummary, EpochIndex)
            tf.logging.info('Step %d: Validation error = %.1f%% (N=%d)' %
                      (EpochIndex, AvgError, ValBatchSize))
            
        # Save the model checkpoint periodically.
        if (EpochIndex % SaveStepInterval == 0 or IsLastEpoch):
            CheckpointPath = os.path.join(TrainDir,'conv.ckpt')
            tf.logging.info('Saving to "%s-%d"', CheckpointPath, EpochIndex)
            Saver.save(Sess, CheckpointPath, global_step=EpochIndex)
            
    tf.logging.info('set_size=%d', TestBatchSize)
    AvgError = 0
    for i in range(0, TestBatchSize, BatchForOneTime):
        Data,EmergencyValueInput=InputReader.ReadBatchData(BatchForOneTime,DisrupRatio,TrainDevice)
        TestSummary, TestError = Sess.run(
            [MergedSummaries, MeanSquaredError],
            feed_dict={
                All0DSignals            : Data['0DSignals'],
                StoredEnergy            : Data['StoredEnergy'],
                Ne_R0                   : Data['Ne_R0'],
                EmergencyValue          : EmergencyValueInput,
                StoredEnergyDropoutProb : 1.0,
                Ne_R0DropoutProb        : 1.0,
                RNNDropoutProb          : 1.0
            })
        AvgError += (TestError*BatchForOneTime)/TestBatchSize
    tf.logging.info('Final test error = %.1f%% (N=%d)' % (AvgError,
                                                           TestBatchSize))
    Sess.close()
