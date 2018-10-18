# author:yellowyellowyao
from __future__ import print_function
import h5py
import numpy as np
import time


def main():
    # "HL-2A":10000-35316
    # "J-TEXT":1038394-1059000
    # "EAST":10000-80000
    device = "HL-2A"
    start_shot = 10000
    end_shot = 35316

    op = classify_shots(start_shot, end_shot, device)


class classify_shots():

    def __init__(self,start_shot,end_shot,device = "HL-2A"):
        shot_numbers = range(start_shot, end_shot)
        self.shot_nums = shot_numbers

        self.no_IP_shots = []
        self.worthless_shots = []
        self.no_flattop_shots = []
        self.saved_shots = []
        self.descend_shots = []
        self.dis_shots = []

        self.save_name = device + "-" + str(start_shot) + "-" + str(end_shot)

        self.min_dis_IP, self.max_dis_IP = 80, 500   # 放电电流最大值位于该区间认为有效放电
        self.end_IP = 25                             # 放电电流小于该值认为是放电结束
        self.discharge_time = 400                    # 放电时间大于该值认为是有效放电

        self.rise_time = 200                         # 电流上升时间
        self.flat_win = 5                            # 判断是否该点平滑的时间窗口
        self.flat_std = 0.2                          # 判断是否该点平滑的标准偏差上限
        self.flat_rate = 0.6                         # 平滑点占排除电流上升时间后放电时间比重

        self.dis_win, self.des_rate = 10, 0.40       # win_len窗口内下降des_rate *IP_max认为发生大破裂

        self.saved_time = 40                         # 大破裂saved_time内未降到end_IP认为被挽回

        self.dis_IP = 0.8                            # 平顶段破裂最小值比例
        self.des_win = 15                            # 小破裂与大破裂之间允许的平台时间
        pass                                         # 低于dis_IP * IP_max,且des_win前未发生小破裂认为是下降段破裂

        self.run()

    def run(self):
        print("dealing....", end=" ")
        for shot_num in self.shot_nums:
            self.classify_one_shot(shot_num)

        configure = ["min_dis_IP, max_dis_IP, send_IP,     discharge_time,"
                     "rise_time,  flat_win,   flat_std,    flat_rate,"
                     "dis_win,    des_rate,   saved_time,  dis_IP,   des_win",
                     self.min_dis_IP, self.max_dis_IP, self.end_IP,     self.discharge_time,
                     self.rise_time,  self.flat_win,   self.flat_std,   self.flat_rate,
                     self.dis_win,    self.des_rate,   self.saved_time, self.dis_IP, self.des_win]

        print("done")

        print("no_IP_shots:", len(self.no_IP_shots))
        print("worthless_shots:", len(self.worthless_shots))
        print("no_flattop_shots:", len(self.no_flattop_shots))
        print("saved_shots:", len(self.saved_shots))
        print("descend_shots:", len(self.descend_shots))
        print("dis_shots:", len(self.dis_shots))
        np.savez(self.save_name, no_IP_shots=self.no_IP_shots,
                 dis_shots=self.dis_shots, saved_shots=self.saved_shots,
                 descend_shots=self.descend_shots, worthless_shots=self.worthless_shots,
                 no_flattop_shots=self.no_flattop_shots, configure=configure)


    def classify_one_shot(self,shot_num):
        try:
            Time, IP = read_hdf5_data(shot_num)
        except KeyError or IndexError:
            self.no_IP_shots.append(shot_num)
            return
        except OSError:
            return

        if not self.worth_shots(IP):
            self.worthless_shots.append(shot_num)
            return

        IP_max, pos_max, pos_0 = self.worth_shots(IP)

        if not self.disruption(IP, IP_max, pos_0):
            return

        max_win, pos_win_max = self.disruption(IP, IP_max, pos_0)

        if not self.have_flattop(IP, pos_0):
            self.no_flattop_shots.append(shot_num)
            return

        if self.saved_dis(pos_0, pos_win_max):
            self.saved_shots.append(shot_num)
            return

        if self.descend(IP, IP_max, max_win, pos_win_max):
            self.descend_shots.append(shot_num)
            return

        self.dis_shots.append(shot_num)



    def worth_shots(self, IP):
        # 起始点流异常
        if IP[0] > self.end_IP:
            return False

        # 最大电流必须位于区间：[min_dis_IP ,max_dis_IP] (否则认为采集错误)
        IP_max = max(IP)
        if IP_max < self.min_dis_IP or IP_max > self.max_dis_IP:
            return False

        # 放电结束时间必须大于discharge_time
        # 放电结束时间：最大电流后降到end_IP位置
        # except:电流无法降到end_IP
        pos_max = np.where(IP == IP_max)[0][0]
        try:
            pos_0 = int(np.where(IP[pos_max:] <= self.end_IP)[0][0]) + pos_max
            if pos_0 < self.discharge_time:
                return False
        except IndexError:
            return False

        # 排除直方图电流和正弦电流等异常信号
        # 判断是否为直方图的窗口时长:10ms
        pos_max_hist = np.where(IP == IP_max)[0][-1]
        if (np.std(IP[pos_max:pos_max_hist]) == 0 and pos_max_hist - pos_max > 10) \
                or np.where(IP <= -0.8 * IP_max)[0].shape[0] > 4:
                return False

        return IP_max,pos_max,pos_0


    def disruption(self,IP, IP_max, pos_0):
        for t in range(self.rise_time, pos_0):
            window = IP[t:t + self.dis_win]
            max_win = max(window)
            pos_win_max = np.where(window == max_win)[0][0] + t
            if max_win >= self.min_dis_IP and\
                 max_win - min(window) >= self.des_rate * IP_max:
                return max_win, pos_win_max


    def have_flattop(self, IP, pos_0):
        IP_std = np.array([np.std(IP[ti:ti + self.flat_win]) for ti in range(len(IP) - self.flat_win)])
        rate = len(np.where(IP_std[self.rise_time:pos_0] <= self.flat_std)[0]) / (pos_0 - self.rise_time)
        if rate >= self.flat_rate:
            return True

        return False


    def saved_dis(self,pos_0,pos_win_max):
        if pos_win_max + self.saved_time < pos_0:
            return True

        return False


    def descend(self,IP,IP_max,max_win,pos_win_max):
        win_start = pos_win_max-self.des_win-self.dis_win
        win_end = pos_win_max-self.des_win
        if max_win < self.dis_IP*IP_max and max(IP[win_start:win_end]) < (self.dis_IP+0.1)*IP_max:
            return True
        return False


def read_hdf5_data(shot_number):
    # try:
    dir_name = shot_number // 200 * 200
    # 加载hdf5文件
    Data = h5py.File(r"\\192.168.9.242\hdf\2A/" + str(dir_name) + "/HL-2A=" + str(shot_number) + "=PhysicsDB.H5", "r")
    # Data = h5py.File(r"\\192.168.9.242\hdf\J-TEXT/"+str(dir_name)+"/JTEXT="+str(shot_number) + "=PhysicsDB.h5","r")
    # Data = h5py.File(r"\\192.168.9.242\hdf\EAST/" + str(dir_name) + "/EAST=" + str(shot_number) + "=PhysicsDB.h5", "r")

    # 读取信号
    IP = Data['PlasmaPara']['IP']
    IP = np.array(IP).reshape(IP.shape[0])
    # 读取信号的附加信息
    T_Start = Data['PlasmaPara']['IP'].attrs['T_Start']
    T_Freq = Data['PlasmaPara']['IP'].attrs['T_Freq']
    data_point_num = IP.shape[0]
    Time = np.arange(T_Start, data_point_num) / T_Freq
    return Time, IP

if __name__ == '__main__':
    t_start = time.clock()
    main()
    print("total cost time:", time.clock() - t_start)