# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 16:06:47 2018

This file defines the class DisrupShotsMySqlProccessor, which can operate(mainly
connect, read and write) the mysql database of disruptive shots

The DisrupShotsMySql database saves the information about the disruption. We can 
use the database to search for the shotnum we want, then get the signals by shotnum
from HDF dataserver

@author: Bi Changliu
"""

import pymysql

class CDisrupShotsMySqlProccessor():
    
    '''
    Some mysql database settings are saved here
    input: none
    output: none
    '''
    def __init__(self):
        self.dbConfig={'host':'192.168.9.222',
                       'user':'',
                       'password':''}
        self.DbName='fusion-ai'
        self.TableName='disruption_param_'
        self.ConnectToMySql()
    
    '''
    Connect to mysql database
    This method will be called when creating the class instance
    But the connection might be closed if there is no opertion for a long time
    If so, call this method to connect again
    input: none
    output: none
    '''
    def ConnectToMySql(self):
        self.Conn = pymysql.Connect(host=self.dbConfig['host'],
                               user=self.dbConfig['user'],
                               password=self.dbConfig['password'],
                               db=self.DbName,
                               charset='utf8',
                               cursorclass=pymysql.cursors.DictCursor)
    
    '''
    This method get parameters of a shot and writes them into the mysql database
    input:   ShotNum: The unique id for each shot
             bValid: If the data should be used in the training dataset
             ReasonForNotValid: Why the data is not used. Will be an empty string
                                if this shot is valid
             TFinishRisingStage: The time when the ip rising stage stop
             TShotLength: The time length of the shot
             IpMax: The maximum of ip
             bDisruptive: If this shot is disruptive
             IpDeltaDuringDisruption: The decline amplitude of ip during the disruption
             TDisruptionStart: The time when the disruption start
             TDisruptionStop: The time when the disruption stop
             ReasonForDisruption: The Reason type of the disruption
             Device: Which device is the data belongs to, HL2A, EAST or JTEXT
    output: none
    '''
    def WriteToMySql(self,
                     ShotNum,
                     bValid,
                     ReasonForNotValid,
                     TFinishRisingStage,
                     TShotLength,
                     IpMax,
                     bDisruptive,
                     IpDeltaDuringDisruption,
                     TDisruptionStart,
                     TDisruptionStop,
                     ReasonForDisruption,
                     Device):
        sql = 'INSERT INTO ' + self.TableName + Device
        sql = sql + ' (ShotNum, bValid, ReasonForNotValid, '
        sql = sql + 'TFinishRisingStage, TShotLength, IpMax, '
        sql = sql + 'bDisruptive, IpDeltaDuringDisruption, TDisruptionStart, '
        sql = sql + 'TDisruptionStop, ReasonForDisruption)'
        sql = sql + 'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'
        with self.Conn.cursor() as Cursor:
            Cursor.execute(sql, (ShotNum, 
                                 bValid, 
                                 ReasonForNotValid,
                                 TFinishRisingStage, 
                                 TShotLength, 
                                 IpMax,
                                 bDisruptive, 
                                 IpDeltaDuringDisruption, 
                                 TDisruptionStart,
                                 TDisruptionStop, 
                                 ReasonForDisruption))
            self.Conn.commit()
    
    '''
    This method find the data of the selected shot number. Then edit the selected
    parameter with the new value
    input:   ShotNum: The unique id for each shot
             Device: Which device is the data belongs to, HL2A, EAST or JTEXT
             **ParamDictToReWrite: This input will content all the params (keys
                                and values) to rewrite
    output: none
    '''
    def ReWriteByShotNum(self,ShotNum,Device,**ParamDictToReWrite):
        with self.Conn.cursor() as Cursor:
            for (key,value) in ParamDictToReWrite.items():
                sql = 'UPDATE ' + self.TableName + Device
                sql =sql + ' SET ' + key + ' = %s WHERE ShotNum = %s'
                Cursor.execute(sql, (value, ShotNum))
        self.Conn.commit()
    
    '''
    This method goes to the table of the selected device and get data of the
    selected shot number
    input:   ShotNum: the shot number of the data you want
             Device: the device which the data belongs to
    output: DisruptionParamDict: a dictionary which contents the wanted data
    '''
    def ReadByShotNum(self,ShotNum,Device):
        sql = 'Select ShotNum, bValid, ReasonForNotValid, '
        sql = sql + 'TFinishRisingStage, TShotLength, IpMax, '
        sql = sql + 'bDisruptive, IpDeltaDuringDisruption, TDisruptionStart, '
        sql = sql + 'TDisruptionStop, ReasonForDisruption '
        sql = sql + 'FROM ' + self.TableName + Device
        sql = sql + ' WHERE ShotNum = %s'
        with self.Conn.cursor() as Cursor:
            Cursor.execute(sql, ShotNum)
            DisruptionParamDict=Cursor.fetchall()
        return DisruptionParamDict[0]
    
    '''
    This method goes to the table of the selected device and delete the data of
    selected shot number
    input:   ShotNum: the shot number of the data you want
             Device: the device which the data belongs to
    output: none
    '''
    def DeleteByShotNum(self,ShotNum,Device):
        sql = 'DELETE FROM ' + self.TableName + Device
        sql = sql + ' WHERE ShotNum = %s'
        with self.Conn.cursor() as Cursor:
            Cursor.execute(sql, ShotNum)
        self.Conn.commit()
    
    '''
    This method goes to the table of the selected device and get all the shot 
    number which satisfies the given condition
    input:   ShotNum: the shot number of the data you want
             Device: the device which the data belongs to
    output: none
    '''
    def SearchShotNumByGivenConditionStr(self,ConditionStr,Device):
        sql = ' SELECT ShotNum FROM ' + self.TableName + Device
        sql = sql + ' WHERE ' + ConditionStr
        with self.Conn.cursor() as Cursor:
            Cursor.execute(sql)
        re = Cursor.fetchall()
        ShotNumList = [x['ShotNum'] for x in re]
        return ShotNumList
    
    '''
    This method means to provide a method to operate the database by mysql command
    directly, in the case that there might always be some need which can't be reached
    by the previous methods
    input:   OperationStr: The mysql command we want to operate
    output: re: The output of the command
    '''
    def FreeOperator(self, OperationStr):
        with self.Conn.cursor() as Cursor:
            Cursor.execute(OperationStr)
        re = Cursor.fetchall()
        return re
