# author:yellowyellowyao
from __future__ import print_function
import h5py
import numpy as np


def read_hdf5(device,shot_number,channel):

    dir_name = shot_number // 200 * 200

    hl2a = ["HL-2A", "HL2A", "2A"]
    jtext = ["J-TEXT", "JTEXT"]
    east = ["EAST"]
    device = device.upper()

    # 加载hdf5文件
    try:
        if device in hl2a:
            Data = h5py.File(r"\\192.168.9.242\hdf\2A/"+str(dir_name)+"/HL-2A="+str(shot_number)+"=PhysicsDB.H5","r")
        elif device in jtext :
            Data = h5py.File(r"\\192.168.9.242\hdf\J-TEXT/"+str(dir_name)+"/JTEXT="+str(shot_number)+"=PhysicsDB.h5","r")
        elif device in east:
            Data = h5py.File(r"\\192.168.9.242\hdf\EAST/"+str(dir_name)+"/EAST="+str(shot_number)+"=PhysicsDB.h5","r")
    except OSError:
            print('No such file or directory')
            return 0, 0

    # 读取信号
    output= []
    for para in Data:
        try:
            output = Data[para][channel]
            break
        except:
            continue

    if not output:
        print("can't find channel")
        return 0,0

    output = np.array(output)

    # 读取信号的附加信息
    T_Start = Data[para][channel].attrs['T_Start']
    T_Freq = Data[para][channel].attrs['T_Freq']
    data_point_num = output.shape[0]
    Time = np.arange(T_Start, data_point_num) / T_Freq

    return Time, output


#################### example ####################
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# Time, Channel = read_hdf5("2a",10022,"IP")
# plt.gcf().set_size_inches(16, 8)
# plt.plot(Time, Channel)
# plt.grid(True)
# plt.show()
