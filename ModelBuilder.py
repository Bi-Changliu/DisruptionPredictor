# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 09:20:12 2018
This file builds CNN for each signal besides 0DSignals and RNN for AllSignals.
The models will be used in the TrainController

@author: Bi ChangLiu
"""

import tensorflow as tf
from SettingsFile import SettingsDict

def LoadVariablesFromCheckpoint(Sess, StartCheckpoint):
    """Utility function to centralize checkpoint restoration.
    
    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(Sess, StartCheckpoint)

def CNNForStoredEnergy(StoredEnergy,IsTraining):
    with tf.name_scope('CNNForStoredEnergy'):
        Param=SettingsDict['ModelParamDict']['CNNForStoredEnergy']
        InputShape=StoredEnergy.get_shape()
        TimeStepSize=InputShape[1]
        """============================reshape输入=============================="""
        if IsTraining:
            DropoutProb = tf.placeholder(tf.float32, name='StoredEnergyDropoutProb')
        InputLength = SettingsDict['InputParamDict']['SignalDict']['StoredEnergy']['FreqDoub']
        FingerprintInput = tf.reshape(StoredEnergy,[-1, InputLength, 1],name='StoredEnergyFingerPrintInput')
        
        """=============================第一卷积层==============================="""
        Conv1KernelLength = Param['Conv1KernelLength']
        Conv1KernelNum = Param['Conv1KernelNum']
    
        Conv1Weights = tf.Variable(
           tf.truncated_normal(
                [Conv1KernelLength, 1, Conv1KernelNum],
                stddev=0.01))
        Conv1Bias = tf.Variable(tf.zeros([Conv1KernelNum]))
        Conv1Output = tf.nn.conv1d(FingerprintInput, Conv1Weights, 1, 'SAME') + Conv1Bias
                                
        Conv1OutputRelu = tf.nn.relu(Conv1Output)
      
        if IsTraining:
            Conv1OutputDropout = tf.nn.dropout(Conv1OutputRelu, DropoutProb)
        else:
            Conv1OutputDropout = Conv1OutputRelu
          
        Pooling1Length = Param['Pooling1Length']
        Pooling1Stride = Param['Pooling1Stride']
        Conv1OutputPooling = tf.nn.pool(input = Conv1OutputDropout, 
                                        window_shape = [Pooling1Length],  
                                        strides = [Pooling1Stride],
                                        pooling_type = "MAX",
                                        padding = 'SAME')
        
        """=============================第二卷积层==============================="""
        Conv2KernelLength = Param['Conv2KernelLength']
        Conv2KernelNum = Param['Conv2KernelNum']
    
        Conv2Weights = tf.Variable(
           tf.truncated_normal(
                [Conv2KernelLength, Conv1KernelNum, Conv2KernelNum],
                stddev=0.01))
        Conv2Bias = tf.Variable(tf.zeros([Conv2KernelNum]))
        Conv2Output = tf.nn.conv1d(Conv1OutputPooling, Conv2Weights, 1, 'SAME') + Conv2Bias
                                
        Conv2OutputRelu = tf.nn.relu(Conv2Output)
      
        if IsTraining:
            Conv2OutputDropout = tf.nn.dropout(Conv2OutputRelu, DropoutProb)
        else:
            Conv2OutputDropout = Conv1OutputRelu
          
        Pooling2Length = Param['Pooling2Length']
        Pooling2Stride = Param['Pooling2Stride']
        Conv2OutputPooling = tf.nn.pool(input = Conv2OutputDropout, 
                                        window_shape = [Pooling2Length],  
                                        strides = [Pooling2Stride],
                                        pooling_type = "MAX",
                                        padding = 'SAME')
    
        """============================第一全连接层=============================="""
        Conv2Shape = Conv2OutputPooling.get_shape()
        Conv2OutputPoolingLength  = Conv2Shape[1]
        Conv2OutputPoolingDimensions = int(Conv2OutputPoolingLength * Conv2KernelNum)
        Con2OutputFlattened = tf.reshape(Conv2OutputPooling,
                                           [-1, Conv2OutputPoolingDimensions])
      
        FC1CellNum=Param['FC1CellNum']
        FC1Weights = tf.Variable(tf.truncated_normal(
                [Conv2OutputPoolingDimensions, FC1CellNum], stddev=0.01))
        FC1Bias = tf.Variable(tf.zeros([FC1CellNum]))
        FC1Output = tf.matmul(Con2OutputFlattened, FC1Weights) + FC1Bias
        
        """============================第二全连接层=============================="""
        FC2CellNum = Param['FC2CellNum']
        FC2Weights = tf.Variable(tf.truncated_normal(
                [FC1CellNum, FC2CellNum], stddev=0.01))
        FC2Bias = tf.Variable(tf.zeros([FC2CellNum]))
        FC2Output = tf.matmul(FC1Output, FC2Weights) + FC2Bias
        
        FC2OutputReshape = tf.reshape(FC2Output,[-1, TimeStepSize, FC2CellNum])
      
        if IsTraining:
          return FC2OutputReshape, DropoutProb
        else:
          return FC2OutputReshape
    
def CNNForNe_R0(StoredEnergy,IsTraining):
    with tf.name_scope('CNNForNe_R0'):
        Param=SettingsDict['ModelParamDict']['CNNForNe_R0']
        InputShape=StoredEnergy.get_shape()
        TimeStepSize=InputShape[1]
        """============================reshape输入=============================="""
        if IsTraining:
            DropoutProb = tf.placeholder(tf.float32, name='DropoutProb')
        InputLength = SettingsDict['InputParamDict']['SignalDict']['Ne_R0']['FreqDoub']
        FingerprintInput = tf.reshape(StoredEnergy,[-1, InputLength, 1])
        
        """=============================第一卷积层==============================="""
        Conv1KernelLength = Param['Conv1KernelLength']
        Conv1KernelNum = Param['Conv1KernelNum']
    
        Conv1Weights = tf.Variable(
           tf.truncated_normal(
                [Conv1KernelLength, 1, Conv1KernelNum],
                stddev=0.01))
        Conv1Bias = tf.Variable(tf.zeros([Conv1KernelNum]))
        Conv1Output = tf.nn.conv1d(FingerprintInput, Conv1Weights, 1, 'SAME') + Conv1Bias
                                
        Conv1OutputRelu = tf.nn.relu(Conv1Output)
      
        if IsTraining:
            Conv1OutputDropout = tf.nn.dropout(Conv1OutputRelu, DropoutProb)
        else:
            Conv1OutputDropout = Conv1OutputRelu
          
        Pooling1Length = Param['Pooling1Length']
        Pooling1Stride = Param['Pooling1Stride']
        Conv1OutputPooling = tf.nn.pool(input = Conv1OutputDropout, 
                                        window_shape = [Pooling1Length],  
                                        strides = [Pooling1Stride],
                                        pooling_type = "MAX",
                                        padding = 'SAME')
        
        """=============================第二卷积层==============================="""
        Conv2KernelLength = Param['Conv2KernelLength']
        Conv2KernelNum = Param['Conv2KernelNum']
    
        Conv2Weights = tf.Variable(
           tf.truncated_normal(
                [Conv2KernelLength, Conv1KernelNum, Conv2KernelNum],
                stddev=0.01))
        Conv2Bias = tf.Variable(tf.zeros([Conv2KernelNum]))
        Conv2Output = tf.nn.conv1d(Conv1OutputPooling, Conv2Weights, 1, 'SAME') + Conv2Bias
                                
        Conv2OutputRelu = tf.nn.relu(Conv2Output)
      
        if IsTraining:
            Conv2OutputDropout = tf.nn.dropout(Conv2OutputRelu, DropoutProb)
        else:
            Conv2OutputDropout = Conv1OutputRelu
          
        Pooling2Length = Param['Pooling2Length']
        Pooling2Stride = Param['Pooling2Stride']
        Conv2OutputPooling = tf.nn.pool(input = Conv2OutputDropout, 
                                        window_shape = [Pooling2Length],  
                                        strides = [Pooling2Stride],
                                        pooling_type = "MAX",
                                        padding = 'SAME')
    
        """============================第一全连接层=============================="""
        Conv2Shape = Conv2OutputPooling.get_shape()
        Conv2OutputPoolingLength  = Conv2Shape[1]
        Conv2OutputPoolingDimensions = int(Conv2OutputPoolingLength * Conv2KernelNum)
        Con2OutputFlattened = tf.reshape(Conv2OutputPooling,
                                           [-1, Conv2OutputPoolingDimensions])
      
        FC1CellNum=Param['FC1CellNum']
        FC1Weights = tf.Variable(tf.truncated_normal(
                [Conv2OutputPoolingDimensions, FC1CellNum], stddev=0.01))
        FC1Bias = tf.Variable(tf.zeros([FC1CellNum]))
        FC1Output = tf.matmul(Con2OutputFlattened, FC1Weights) + FC1Bias
        
        """============================第二全连接层=============================="""
        FC2CellNum = Param['FC2CellNum']
        FC2Weights = tf.Variable(tf.truncated_normal(
                [FC1CellNum, FC2CellNum], stddev=0.01))
        FC2Bias = tf.Variable(tf.zeros([FC2CellNum]))
        FC2Output = tf.matmul(FC1Output, FC2Weights) + FC2Bias
      
        FC2OutputReshape = tf.reshape(FC2Output,[-1, TimeStepSize, FC2CellNum])
      
        if IsTraining:
          return FC2OutputReshape, DropoutProb
        else:
          return FC2OutputReshape

def RNNForAllSignals(AllSignals):
    with tf.name_scope('RNNForAllSignals'):
        LayerNum=SettingsDict['ModelParamDict']['RNNForAllSignals']['LayerNum']
        HiddenSize=SettingsDict['ModelParamDict']['RNNForAllSignals']['HiddenSize']
        DropoutProb = tf.placeholder(tf.float32, name='DropoutProb')
        BatchSize = SettingsDict['TrainParamDict']['BatchNum']['BatchForOneTime']
        TimeLength = SettingsDict['InputParamDict']['MinDataLen']
        LSTMCell = tf.contrib.rnn.BasicLSTMCell(num_units=HiddenSize, 
                                                        forget_bias=1.0, 
                                                        state_is_tuple=True)
        LSTMCellDropout = tf.contrib.rnn.DropoutWrapper(cell=LSTMCell, 
                                                         input_keep_prob=1.0, 
                                                         output_keep_prob=DropoutProb)
        MultiLSTMCell = tf.contrib.rnn.MultiRNNCell([LSTMCellDropout] * LayerNum, 
                                                            state_is_tuple=True)
        InitState = MultiLSTMCell.zero_state(BatchSize, dtype=tf.float32)
        
        RNNOutput, State = tf.nn.dynamic_rnn(MultiLSTMCell, 
                                           inputs=AllSignals, 
                                           initial_state=InitState, 
                                           time_major=False)
        FingerPrintInput = tf.reshape(RNNOutput,[-1,HiddenSize])
        """============================第一全连接层=============================="""
      
        FC1CellNum=SettingsDict['ModelParamDict']['RNNForAllSignals']['FC1CellNum']
        FC1Weights = tf.Variable(tf.truncated_normal(
                [HiddenSize, FC1CellNum], stddev=0.01))
        FC1Bias = tf.Variable(tf.zeros([FC1CellNum]))
        FC1Output = tf.matmul(FingerPrintInput, FC1Weights) + FC1Bias
        
        """============================第二全连接层=============================="""
        FC2CellNum = SettingsDict['ModelParamDict']['RNNForAllSignals']['FC2CellNum']
        FC2Weights = tf.Variable(tf.truncated_normal(
                [FC1CellNum, FC2CellNum], stddev=0.01))
        FC2Bias = tf.Variable(tf.zeros([FC2CellNum]))
        FC2Output = tf.matmul(FC1Output, FC2Weights) + FC2Bias
      
        FC2OutputReshape = tf.reshape(FC2Output,[-1, TimeLength])
      
        return FC2OutputReshape,DropoutProb