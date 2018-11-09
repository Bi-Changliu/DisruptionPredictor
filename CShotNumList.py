# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:17:41 2018

This file defines the ShotNumList class, which contains all the valid shot numbers
in mysql and can give a batch of shot numbers by given condition

@author: Bi ChangLiu
"""
import MysqlDatabaseUtils
import numpy as np

class CShotNumList(object):
    
    def __init__(self,Device):
        self.Device=Device
        DisrupShotsMySqlProccessor=MysqlDatabaseUtils.CDisrupShotsMySqlProccessor()
        ConditionStr='bValid=1 AND bDisruptive=0'
        self.SafeShotNumbers=np.array(DisrupShotsMySqlProccessor.SearchShotNumByGivenConditionStr(ConditionStr,Device))
        ConditionStr='bValid=1 AND bDisruptive=1'
        self.DisrupShotNumbers=np.array(DisrupShotsMySqlProccessor.SearchShotNumByGivenConditionStr(ConditionStr,Device))
        
    def GetShotNumBatch(self,BatchSize,DisrupRatio):
        assert DisrupRatio>=0
        assert DisrupRatio<=1
        SafeSetSize=len(self.SafeShotNumbers)
        SafeBatchSize=BatchSize-int(BatchSize*DisrupRatio)
        SafeRandomIdxSet=np.random.permutation(SafeSetSize)[:SafeBatchSize]
        DisrupSetSize=len(self.DisrupShotNumbers)
        DisrupBatchSize=int(BatchSize*DisrupRatio)
        DisrupRandomIdxSet = np.random.permutation(DisrupSetSize)[:DisrupBatchSize]
        BatchShotNums=np.append(self.DisrupShotNumbers[DisrupRandomIdxSet],self.SafeShotNumbers[SafeRandomIdxSet])
        np.random.shuffle(BatchShotNums)
        return BatchShotNums
    
    def GetShotNumByRangeAndRatio(self,StartShotNum,StopShotNum,DisrupRatio=None):
        SafeShotNumbersInRange=self.SafeShotNumbers[np.where(self.SafeShotNumbers>StartShotNum)]
        SafeShotNumbersInRange=SafeShotNumbersInRange[np.where(SafeShotNumbersInRange<StopShotNum)]
        SafeSetSize=len(SafeShotNumbersInRange)
        DisrupShotNumbersInRange=self.DisrupShotNumbers[np.where(self.DisrupShotNumbers>StartShotNum)]
        DisrupShotNumbersInRange=DisrupShotNumbersInRange[np.where(DisrupShotNumbersInRange<StopShotNum)]
        DisrupSetSize=len(DisrupShotNumbersInRange)
        if not DisrupRatio is None:
            assert DisrupRatio>=0
            assert DisrupRatio<=1
            if (SafeSetSize/(1-DisrupRatio)>DisrupSetSize/DisrupRatio):
                DisrupBatch=DisrupShotNumbersInRange
                SafeBatchSize=int(DisrupSetSize/DisrupRatio*(1-DisrupRatio))
                SafeRandomIdxSet=np.random.permutation(SafeSetSize)[:SafeBatchSize]
                SafeBatch=SafeShotNumbersInRange[SafeRandomIdxSet]
            else:
                SafeBatch=SafeShotNumbersInRange
                DisrupBatchSize=int(SafeSetSize*DisrupRatio/(1-DisrupRatio))
                DisrupRandomIdxSet=np.random.permutation(DisrupSetSize)[:DisrupBatchSize]
                DisrupBatch=DisrupShotNumbersInRange[DisrupRandomIdxSet]
        else:
            DisrupBatch=DisrupShotNumbersInRange
            SafeBatch=SafeShotNumbersInRange   
        BatchShotNums=np.append(SafeBatch,DisrupBatch)
        np.random.shuffle(BatchShotNums)
        return BatchShotNums
        
ShotNumList=CShotNumList('HL2A')
BatchShotNums=ShotNumList.GetShotNumByRangeAndRatio(20000,35000,0.5)