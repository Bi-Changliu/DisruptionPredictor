# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:48:47 2018
This file controls the training

@author: Bi ChangLiu
"""
import os
import numpy as np
import tensorflow as tf
import ModelBuilder
import TfRecordReader
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

GpuNum=SettingsDict['TrainParamDict']['GpuNum']
DisrupRatio=SettingsDict['TrainParamDict']['DisrupRatio']

tf.reset_default_graph()
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as Sess:
    DropoutProb = tf.placeholder(tf.float32, name='DropoutProb')
    
    TrainInputBatch,TrainEmergencyValue=TfRecordReader.ReadAndDecodeTfRecords('Train',BatchForOneTime)
    ValInputBatch,ValEmergencyValue=TfRecordReader.ReadAndDecodeTfRecords('Val',BatchForOneTime)
    bValidation=tf.Variable(False)
    EmergencyValue3D = tf.cond(bValidation, lambda: tf.identity(TrainEmergencyValue), lambda: tf.identity(ValEmergencyValue))
    EmergencyValue = tf.reshape(EmergencyValue3D,[BatchForOneTime,MinDataLen])
    TrainAll0DSignals = TrainInputBatch['0DSignals']
    ValAll0DSignals = ValInputBatch['0DSignals']
    All0DSignals = tf.cond(bValidation, lambda: tf.identity(TrainAll0DSignals), lambda: tf.identity(ValAll0DSignals))
    TrainStoredEnergy = TrainInputBatch['StoredEnergy']
    ValStoredEnergy = ValInputBatch['StoredEnergy']
    StoredEnergy = tf.cond(bValidation, lambda: tf.identity(TrainStoredEnergy), lambda: tf.identity(ValStoredEnergy))
    TrainNe_R0 = TrainInputBatch['Ne_R0']
    ValNe_R0 = ValInputBatch['Ne_R0']
    Ne_R0 = tf.cond(bValidation, lambda: tf.identity(TrainNe_R0), lambda: tf.identity(ValNe_R0))
    TrainMir_Pol_A = TrainInputBatch['Mir_Pol_A']
    ValMir_Pol_A = ValInputBatch['Mir_Pol_A']
    Mir_Pol_A = tf.cond(bValidation, lambda: tf.identity(TrainMir_Pol_A), lambda: tf.identity(ValMir_Pol_A))
    
    All0DSignalsSplits=tf.split(All0DSignals,GpuNum)
    StoredEnergySplits=tf.split(StoredEnergy,GpuNum)
    Ne_R0Splits=tf.split(Ne_R0,GpuNum)
    Mir_Pol_ASplits=tf.split(Mir_Pol_A,GpuNum)
    EmergencyValueSplits=tf.split(EmergencyValue,GpuNum)
    MeanSquaredErrorTower=[]
    RNNOutputMeanTower=[]
    Ne_R0CNNOutputMeanTower=[]
    
    with tf.variable_scope(tf.get_variable_scope()):
        for GpuIndex in range(GpuNum):
            with tf.device('/GPU:%s' % GpuIndex):
                StoredEnergyCNNOutput = ModelBuilder.CNNForStoredEnergy(StoredEnergySplits[GpuIndex], DropoutProb, IsTraining=True)
                Ne_R0CNNOutput = ModelBuilder.CNNForNe_R0(Ne_R0Splits[GpuIndex], DropoutProb, IsTraining=True)
                Mir_Pol_ACNNOutput = ModelBuilder.CNNForMir_Pol_A(Mir_Pol_ASplits[GpuIndex], DropoutProb, IsTraining=True)
                
                AllSignalsSplits = tf.concat([All0DSignalsSplits[GpuIndex],
                                              StoredEnergyCNNOutput,
                                              Ne_R0CNNOutput,
                                              Mir_Pol_ACNNOutput,
                                              ],2,name='AllSignalsSplits')
                
                RNNOutput=ModelBuilder.RNNForAllSignals(AllSignalsSplits ,DropoutProb)
                
                MeanSquaredErrorSplit = tf.losses.mean_squared_error(EmergencyValueSplits[GpuIndex], RNNOutput)
                RNNOutputMeanSplit = tf.reduce_mean(RNNOutput)
                Ne_R0CNNOutputMeanSplit = tf.reduce_mean(Ne_R0CNNOutput)
                
                MeanSquaredErrorTower.append(MeanSquaredErrorSplit)
                RNNOutputMeanTower.append(RNNOutputMeanSplit)
                Ne_R0CNNOutputMeanTower.append(Ne_R0CNNOutputMeanSplit)
                tf.get_variable_scope().reuse_variables()
    
    # Create the back propagation and training evaluation machinery in the graph.
    with tf.name_scope('MeanSquaredError'):
        MeanSquaredError=tf.reduce_mean(MeanSquaredErrorTower)
        RNNOutputMean=tf.reduce_mean(RNNOutputMeanTower)
        Ne_R0CNNOutputMean=tf.reduce_mean(Ne_R0CNNOutputMeanTower)
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
    Coord=tf.train.Coordinator()
    Threads=tf.train.start_queue_runners(sess=Sess, coord=Coord)
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

        # Pull the audio samples we'll use for training.
        TrainSummary, TrainError, TrainRNNOutputMean, TrainNe_R0CNNOutputMean ,_, _ = Sess.run(
            [
                MergedSummaries, MeanSquaredError, RNNOutputMean, Ne_R0CNNOutputMean, TrainStep,
                IncrementGlobalStep
            ],
            feed_dict={
                LearningRate            : LearningRateInput,
                DropoutProb             : 0.5,
            })
        TrainWriter.add_summary(TrainSummary, EpochIndex)
        tf.logging.info('Step #%d: rate %f, error %.1f, outputmean %d, outputmean2 %f' %
                    (EpochIndex, LearningRateInput, TrainError,
                     TrainRNNOutputMean,TrainNe_R0CNNOutputMean))
        IsLastEpoch=(EpochIndex==EpochNum)
        
        #经过一定的training step就validate一次
        if (EpochIndex % EvalStepInterval)==0 or IsLastEpoch:
            AvgError = 0
            for i in range(0, ValBatchSize, BatchForOneTime):
                ValSummary, ValError, ValRNNOutputMean = Sess.run(
                    [MergedSummaries, MeanSquaredError, RNNOutputMean],
                    feed_dict={
                        DropoutProb : 1.0,
                    })
                AvgError += (ValError*BatchForOneTime)/ValBatchSize
            ValidationWriter.add_summary(ValSummary, EpochIndex)
            tf.logging.info('Step %d: Validation error = %.1f (N=%d), outputmean %d' %
                      (EpochIndex, AvgError, ValBatchSize,
                     ValRNNOutputMean))
            
        # Save the model checkpoint periodically.
        if (EpochIndex % SaveStepInterval == 0 or IsLastEpoch):
            CheckpointPath = os.path.join(TrainDir,'conv.ckpt')
            tf.logging.info('Saving to "%s-%d"', CheckpointPath, EpochIndex)
            Saver.save(Sess, CheckpointPath, global_step=EpochIndex)
            
    tf.logging.info('set_size=%d', TestBatchSize)
    AvgError = 0
    for i in range(0, TestBatchSize, BatchForOneTime):
        TestSummary, TestError = Sess.run(
            [MergedSummaries, MeanSquaredError],
            feed_dict={
                DropoutProb : 1.0,
            })
        AvgError += (TestError*BatchForOneTime)/TestBatchSize
    tf.logging.info('Final test error = %.1f (N=%d)' % (AvgError,
                                                           TestBatchSize))
    Coord.request_stop()
    Coord.join(Threads)
    Sess.close()
