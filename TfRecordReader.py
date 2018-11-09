# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:26:28 2018

This program is used to read and decode the data

@author: zy-yang
"""
import tensorflow as tf
from SettingsFile import SettingsDict

def ReadAndDecodeTfRecords(Mode,BatchSize):
    assert Mode in ['Train','Val','Test']
    MinDataLen=SettingsDict['InputParamDict']['MinDataLen']
    TfRecordsDir=SettingsDict['TrainParamDict']['TfRecordsDir']
    
    TfRecordsPathName=TfRecordsDir+'/'+Mode+'Data.tfrecords'
    FileQueue = tf.train.string_input_producer([TfRecordsPathName])
    Reader = tf.TFRecordReader()
    _,TfSerialized = Reader.read(FileQueue)
    
    FeaturesDict={
        "StoredEnergy":tf.FixedLenFeature([],tf.string),
        "Ne_R0":tf.FixedLenFeature([],tf.string),
        "0DSignals":tf.FixedLenFeature([],tf.string),
        "EmergencyValue":tf.FixedLenFeature([],tf.string)}
    TfFeature = tf.parse_single_example(TfSerialized,features=FeaturesDict)
    
    SignalDict=SettingsDict['InputParamDict']['SignalDict']
    StoredEnergy = tf.decode_raw(TfFeature["StoredEnergy"],tf.float64)
    StoredEnergy = tf.reshape(StoredEnergy,[MinDataLen,SignalDict['StoredEnergy']['FreqDoub']])
    Ne_R0 = tf.decode_raw(TfFeature["Ne_R0"],tf.float64)
    Ne_R0 = tf.reshape(Ne_R0,[MinDataLen,SignalDict['Ne_R0']['FreqDoub']])
    Mir_Pol_A = tf.decode_raw(TfFeature["Mir_Pol_A"],tf.float64)
    Mir_Pol_A = tf.reshape(Mir_Pol_A,[MinDataLen,SignalDict['Mir_Pol_A']['FreqDoub']])
    ZeroDSignals = tf.decode_raw(TfFeature["0DSignals"],tf.float64)
    ZeroDSignals = tf.reshape(ZeroDSignals,[MinDataLen,SettingsDict['InputParamDict']['0DSignalNum']])
    EmergencyValue = tf.decode_raw(TfFeature["EmergencyValue"],tf.float64)
    EmergencyValue = tf.reshape(EmergencyValue,[MinDataLen,1])
    
    StoredEnergyBatch,Ne_R0Batch,ZeroDSignalsBatch,EmergencyValueBatch = \
    tf.train.shuffle_batch([StoredEnergy,Ne_R0,ZeroDSignals,EmergencyValue],
                                                   batch_size=BatchSize,
                                                   capacity=BatchSize*10,
                                                   min_after_dequeue=BatchSize*5)
    Data=dict()
    Data['StoredEnergy'] = tf.cast(StoredEnergyBatch, tf.float32)
    Data['Ne_R0'] = tf.cast(Ne_R0Batch, tf.float32)
# =============================================================================
#     Data['P_Rad_Bolometer'] = tf.cast(P_Rad_Bolometer, tf.float32)
#     Data['H_Mode_Ha'] = tf.cast(H_Mode_Ha, tf.float32)
#     Data['VLoop'] = tf.cast(VLoop, tf.float32)
# =============================================================================
    Data['0DSignals'] = tf.cast(ZeroDSignalsBatch, tf.float32)
    EmergencyValue = tf.cast(EmergencyValueBatch, tf.float32)
    
    return Data,EmergencyValue

tf.reset_default_graph()
with tf.Session() as Sess:
    Data,EmergencyValue=ReadAndDecodeTfRecords('Train',20)
    tf.global_variables_initializer().run()
    Coord=tf.train.Coordinator()
    Threads=tf.train.start_queue_runners(sess=Sess, coord=Coord)
    a,b=Sess.run([Data,EmergencyValue])
    Coord.request_stop()
    Coord.join(Threads)
    Sess.close()
    