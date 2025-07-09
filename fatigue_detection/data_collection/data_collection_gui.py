import numpy as np
from psychopy import visual, event, core, monitors
from psychopy.visual import ButtonStim
from .data_logger import DataLogger
from .fatigue_prompt import FatiguePromptUI
from .label_viewer import LabelViewerUI
from .subject_input import SubjectNameInputUI
from .fatigue_detection_paradigm import FatigueDetectionParadigm
from metabci.brainstim.framework import Experiment
from .nd_device_demo import tcp_test
import time
import math
import os
from scipy.io import savemat
from neuro_dance.nd_device_process import NdDeviceBase
from neuro_dance.linkedlist import NdLinkedlist

class DataCollectionUI:
    def __init__(self, win):
        self.win = win
        self.return_to_main = False  # 标记是否返回主界面
        self.subject_name = None  # 存储输入的被试名称
        self.connect = 0
        self.init_ui()
        # 每秒钟接收的数据包数量
        self.package_per_second = 5
        # 存储5分钟的数据，计算存储的包数量
        self.eeg_package_count = self.package_per_second * 60 * 5  # 脑电图数据包数量
        self.eog_package_count = self.package_per_second * 60 * 5  # 眼电图数据包数量

    def init_ui(self):
        self.title_text = visual.TextStim(self.win, text="数据采集", pos=(0, 0.7), color="white",
                                          height=0.1, font='Microsoft YaHei')
        # 创建按钮
        self.start_button = ButtonStim(self.win, text='开始采集', pos=(0, -0.3), size=(0.5, 0.1), font='Microsoft YaHei')
        self.back_button = ButtonStim(self.win, text="主页", pos=(-0.7, 0.7), size=(0.25, 0.08), font='Microsoft YaHei')
        self.label_button = ButtonStim(self.win, text="标签列表", pos=(0, 0.3), size=(0.5, 0.1), font='Microsoft YaHei')
        self.connect_button = ButtonStim(self.win, text='连接设备', pos=(0, 0), size=(0.5, 0.1), font='Microsoft YaHei')
        # 显示文件路径的文本框
        self.connect_check = visual.TextStim(self.win, text="设备未连接",
                                              pos=(0.45, 0), color="white", height=0.04, font='Microsoft YaHei')

    def show(self):
        mouse = event.Mouse(win=self.win)

        while True:
            # 绘制所有组件
            self.title_text.draw()
            self.start_button.draw()
            self.back_button.draw()
            self.label_button.draw()
            self.connect_button.draw()
            self.connect_check.draw()
            self.win.flip()

            # 检测按键
            keys = event.getKeys()
            if 'escape' in keys:
                self.return_to_main = True
                break

            # 检测鼠标点击
            if mouse.isPressedIn(self.connect_button):
                try:
                    self.nd_device = self.NdDevice(self.eeg_package_count, self.eog_package_count, mode='tcp', com='', tcp_ip='192.168.0.111', tcp_port=8899, host_mac_bytes=None)
                    self.nd_device.start()  # 启动设备

                    index=0
                    while index < 100:  # 循环10次
                        time.sleep(0.1)  # 每0.1秒读取一次数据

                        index = index + 1
                        millis_second = int(round(time.time() * 1000))  # 获取当前时间戳（毫秒）
                        time_span = 1000  # 读取过去1秒的数据
                        read_data = self.nd_device.read_latest_eeg_data()  # 读取数据
                        if read_data is not None:
                            print(read_data.shape)
                            # nd_device.close()
                            self.connect = 1
                            self.connect_check.setText("设备已连接，可以采集数据")
                            break
                except:
                    self.connect = 0
                print('self.connect', self.connect)

            if mouse.isPressedIn(self.start_button):
                print(">>> begin data collection 1 ...")
                # 实例化并注册“伪范式”
                fatigue_paradigm = FatigueDetectionParadigm(win=self.win)

                # 创建实验对象并注册范式（但不真正显示）
                mon = monitors.Monitor(name='primary_monitor', width=59.6, distance=60)
                mon.setSizePix([1920, 1080])
                ex = Experiment(
                    monitor=mon,
                    bg_color_warm=np.array([0, 0, 0]),
                    screen_id=0,
                    win_size=[1920, 1080],
                    is_fullscr=False,
                    record_frames=False,
                    disable_gc=False,
                    process_priority='normal',
                    use_fbo=False,
                )
                # 注册范式（即使没有视觉刺激，也用于标记时间戳或准备后续处理）
                fatigue_paradigm.register_to_experiment(ex)  # 使用新定义的方法

                core.wait(0.2)  # 等待0.2秒
                # self.input_subject_name()  # 输入被试名
                if self.input_subject_name():
                    self.run_data_collection(ex)  # 将实验对象传递给数据采集过程

            elif mouse.isPressedIn(self.back_button):
                self.return_to_main = True
                break
            elif mouse.isPressedIn(self.label_button):
                self.return_to_main = self.show_label_viewer()
                if self.return_to_main:
                    break
        return self.return_to_main   # 返回是否回到主界面的标志

    def run_data_collection(self, ex):
        """数据采集主流程"""
        if self.connect:
            connect_text = '设备已连接，脑电与标签数据采集中。\n目前设置提示框间隔为6s'
        else:
            connect_text = '设备未连接，标签数据采集中。\n目前设置提示框间隔为6s'

        data_list = np.empty((8, 0))  # 初始化一个空的 (8, 0) 数组，用于存eeg数据
        win = self.win
        logger = DataLogger(self.subject_name)
        clock = core.Clock()
        start_time = clock.getTime()  # 记录开始采集时间
        last_prompt_time = start_time  # 记录最后弹出的时间
        prompt_box_interval_time = 6  # 提示框间隔时间， 测试用短时间，正式可用600秒

        status_text = visual.TextStim(
            win,
            text=connect_text,
            color="lime",  # 醒目颜色
            height=0.12,  # 更大字号
            pos=(0, 0),  # 屏幕正中央
            alignText='center',  # 文本居中对齐
            anchorHoriz='center',  # 确保水平居中
            font='Microsoft YaHei'
        )

        end_button = ButtonStim(win, text="完成", pos=(0, -0.3), size=(0.3, 0.1), font='Microsoft YaHei')
        fatigue_prompt = FatiguePromptUI(win)

        mouse = event.Mouse(win=win)
        print(">>> begin data collection...")
        logger.write_start_time()
        while True:
            current_time = clock.getTime()
            if self.connect:
                time.sleep(0.1)
                read_data = self.nd_device.read_latest_eeg_data()  # 读取数据
                # 处理数据格式 - 从(8, N, 1)格式转换为(8, N)
                if read_data is not None and len(read_data.shape) == 3 and read_data.shape[2] == 1:
                    read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
                if read_data is not None:
                    data_list = np.hstack((data_list, read_data))

            # 检查是否点击结束按钮
            if mouse.isPressedIn(end_button):
                if self.connect:
                    # nd_device.close()  # 关闭设备连接
                    save_path = os.path.join(os.path.dirname(__file__), 'data')  # 获取当前文件夹下的 data 目录
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)  # 创建目录
                    data_file_path = os.path.join(save_path, f'EEG_{self.subject_name}.mat')
                    savemat(data_file_path, {'EEG': data_list})
                logger.write_end_time()
                logger.close()
                # 提示用户数据采集结束，即将返回主界面
                ending_text = visual.TextStim(
                    win, text="数据采集完成！\n返回上级菜单...",
                    color="green", height=0.08, pos=(0, 0.3), font='Microsoft YaHei'
                )
                ending_text.draw()
                win.flip()
                core.wait(1.5)  # 显示提示 1.5 秒
                return

            # 每 6 秒弹出提示（可改为 600 秒即 10 分钟）
            if current_time - last_prompt_time >= prompt_box_interval_time:
                level, prompt_start_time, key_press_time = fatigue_prompt.show_prompt()
                logger.write_prompt_shown(prompt_start_time)
                logger.write_user_input(level, key_press_time)

                # 同步发送LSL事件
                if hasattr(ex, 'send_event'):
                    ex.send_event(key='prompt', value=str(level))  # 示例：向LSL发送事件标记

                last_prompt_time = current_time  # 更新上一次弹出时间

            # 实时刷新屏幕
            end_button.draw()
            status_text.draw()
            win.flip()

    def input_subject_name(self):
        input_ui = SubjectNameInputUI(self.win)
        result = input_ui.show()
        if result:
            self.subject_name = input_ui.subject_name
            return True
        else:
            # self.return_to_main = input_ui.return_to_main
            return False

    def show_label_viewer(self):
        viewer = LabelViewerUI(self.win)
        return_to_main = viewer.show()
        return return_to_main

    class NdDevice(NdDeviceBase):
        """
        神经接口设备类，用于与硬件设备通信并处理脑电图和眼电图数据
        继承自NdDeviceBase基类
        """
        # 使用链表存储脑电图和眼电图数据
        eeg_datas = NdLinkedlist()  # 存储脑电图数据的链表
        eog_datas = NdLinkedlist()  # 存储眼电图数据的链表
        eeg_sample = 1000  # 脑电图采样率默认为1000Hz
        mode = None  # 设备通信模式
        last_read_timestamp = 0  # 初始化为0

        def __init__(self, eeg_package_count, eog_package_count, mode, com, tcp_ip, tcp_port, host_mac_bytes=None):
            """
            初始化设备
            :param mode: 通信模式，'serial'为串口模式，'tcp'为网络模式
            :param com: 串口号，串口模式下使用
            :param tcp_ip: TCP服务器IP地址，网络模式下使用
            :param tcp_port: TCP服务器端口，网络模式下使用
            :param host_mac_bytes: 主机MAC地址字节数组，可选
            """
            # super(NdDeviceBase, self).__init__(mode, com, tcp_ip, tcp_port, host_mac_bytes)
            NdDeviceBase.__init__(self, mode, com, tcp_ip, tcp_port, host_mac_bytes)
            self.mode = mode
            self.eeg_package_count = eeg_package_count
            self.eog_package_count = eog_package_count

        def host_info(self):
            """
            获取主机(dongle)信息，包括版本、序列号和MAC地址
            """
            super(self.NdDevice, self).host_version_info()
            time.sleep(0.01)  # 短暂延时，避免通信冲突
            super(self.NdDevice, self).host_sn_info()
            time.sleep(0.01)
            super(self.NdDevice, self).host_mac_info()

        def device_info(self):
            """
            获取设备信息，包括版本、序列号和MAC地址
            """
            super(self.NdDevice, self).device_version_info()
            time.sleep(0.01)  # 短暂延时，避免通信冲突
            super(self.NdDevice, self).device_sn_info()
            time.sleep(0.01)
            super(self.NdDevice, self).device_mac_info()

        def battery(self):
            """
            获取设备电池电量信息
            """
            super(self.NdDevice, self).device_battery()

        def device_pair(self, mac):
            """
            与指定MAC地址的设备配对
            :param mac: 设备MAC地址
            """
            super(self.NdDevice, self).pair(mac)

        # 扫描蓝牙设备
        # 扫描到的设备通过devices_received(devices)回调函数返回
        def devices_scan(self):
            """
            扫描可用的蓝牙设备
            扫描结果通过devices_received回调函数返回
            """
            super(self.NdDevice, self).device_scan()

        def host_mac_received(self, host_mac):
            """
            接收到主机MAC地址的回调函数
            :param host_mac: 主机MAC地址
            """
            print("Test host_mac_received:" + host_mac)

        def host_sn_received(self, host_sn):
            """
            接收到主机序列号的回调函数
            :param host_sn: 主机序列号
            """
            print("Test host_sn_received:" + host_sn)

        def channel_received(self, data):
            """
            接收到通道数据的回调函数
            :param data: 通道数据
            """
            print("Test channel_received:" + data)

        def host_version_received(self, host_version):
            """
            接收到主机版本信息的回调函数
            :param host_version: 主机版本信息
            """
            print("Test host_version_received:" + host_version)

        def device_mac_received(self, device_mac):
            """
            接收到设备MAC地址的回调函数
            :param device_mac: 设备MAC地址
            """
            print("Test device mac:{0}".format(device_mac))

        def device_sn_received(self, device_sn):
            """
            接收到设备序列号的回调函数
            :param device_sn: 设备序列号
            """
            print("Test device_sn_received:{0}".format(device_sn))

        def device_battery_received(self, battery):
            """
            接收到设备电池电量信息的回调函数
            :param battery: 电池电量
            """
            print("Test battery:{0}".format(battery))

        def device_version_received(self, device_version):
            """
            接收到设备版本信息的回调函数
            :param device_version: 设备版本信息
            """
            print("Test device version:{0}".format(device_version))

        def eeg_received(self, data):
            """
            接收到脑电图数据的回调函数
            :param data: 包含时间戳和数据的字典
            """
            shape = self.array_shape(data['data'])
            # print("unix milliseconds(first point time):{0},shape:{1},data:{2}".format(data['timestamp'], shape, data['data']))
            # 如果数据超过设定的存储量，移除最早的数据
            if self.eeg_datas.length() > self.eeg_package_count:
                self.eeg_datas.removeHead()
            self.eeg_datas.add(data)  # 将新数据添加到链表中

        def eog_received(self, data):
            """
            接收到眼电图数据的回调函数
            :param data: 包含时间戳和数据的字典
            """
            shape = self.array_shape(data['data'])
            # print("unix milliseconds(first point time):{0},shape:{1},data:{2}".format(data['timestamp'], shape, data['data']))
            # 如果数据超过设定的存储量，移除最早的数据
            if self.eog_datas.length() > self.eog_package_count:
                self.eog_datas.removeHead()
            self.eog_datas.add(data)  # 将新数据添加到链表中

        def read_latest_eeg_data(self, target_freq=250):
            """
            读取最新的脑电图数据，并按照目标频率进行下采样
            :param target_freq: 目标采样频率（Hz），默认250Hz
            :return: 下采样后的脑电图数据，形状为(8,N)
            """
            current_time = int(round(time.time() * 1000))

            # 设备实际采样率
            device_freq = 1000  # 设备采样率1000Hz

            # 计算下采样比率
            downsample_ratio = int(device_freq / target_freq)  # 1000/250 = 4

            # 获取最新数据包
            latest_packet = None
            latest_time = 0

            for i in range(self.eeg_datas.length()):
                packet = self.eeg_datas.eleAt(i)
                if packet is None:
                    continue

                if packet['timestamp'] > latest_time:
                    latest_time = packet['timestamp']
                    latest_packet = packet

            # 如果找到新数据包且比上次读取的更新
            if latest_packet and latest_time > self.last_read_timestamp:
                self.last_read_timestamp = latest_time

                # 获取原始数据
                raw_data = latest_packet['data']

                # 如果是numpy数组，进行下采样
                if isinstance(raw_data, np.ndarray):
                    # 对每个通道的数据进行下采样（每downsample_ratio个点取1个）
                    downsampled_data = raw_data[:, ::downsample_ratio]
                    return downsampled_data
                else:
                    # 对列表形式的数据进行下采样
                    downsampled_data = []
                    for channel in raw_data:
                        downsampled_channel = channel[::downsample_ratio]
                        downsampled_data.append(downsampled_channel)
                    return np.array(downsampled_data)

            return None

        def __read_eeg_date_tcp(self, start_millis_second, read_millisecond, freq):
            """
            TCP模式下读取指定时间段的脑电图数据
            :param start_millis_second: 起始时间戳（毫秒）
            :param read_millisecond: 要读取的时间长度（毫秒）
            :param freq: 采样频率（Hz）
            :return: 指定时间段的脑电图数据，如果数据不足则返回None
            """
            point_per_millis = freq / 1000  # 每毫秒的数据点数
            eeg_data = None
            while self.eeg_datas.length() > 0:
                eeg_left = self.eeg_datas.eleAt(0)  # 获取链表中第一个数据包
                if eeg_left is None:
                    break
                packet_start_millis = eeg_left['timestamp']  # 数据包的起始时间戳
                packet_end_millis = len(eeg_left['data'][0]) * 1000 / freq + packet_start_millis  # 数据包的结束时间戳
                # 如果数据包结束时间早于请求的起始时间，则移除该数据包
                if packet_end_millis < start_millis_second:
                    self.eeg_datas.removeHead()
                    continue
                # 计算数据起始位置
                eeg_start_position = int((start_millis_second - packet_start_millis) * point_per_millis)
                if eeg_start_position < 0:
                    eeg_start_position = 0
                # 多加10个冗余点，确保数据足够
                need_point_count = int(read_millisecond * point_per_millis)
                check_point_count = need_point_count + eeg_start_position + 10
                # 检查是否有足够的数据点
                enough, channel_data = self.__check_data_enough(self.eeg_datas, check_point_count)
                if enough:
                    # 将多个通道的数据水平拼接
                    eeg_data = np.hstack(channel_data)
                    # 截取指定时间段的数据
                    eeg_data = eeg_data[:, eeg_start_position:(eeg_start_position + need_point_count)]
                    break
            return eeg_data

        def __check_data_enough(self, datas, read_point_count):
            """
            检查是否有足够的数据点
            :param datas: 数据链表
            :param read_point_count: 需要的数据点数
            :return: (是否足够, 数据列表)
            """
            eleIndex = 0
            count = 0
            need_data = []
            # 遍历链表，累计数据点数
            while datas.length() > eleIndex and count < read_point_count:
                ele = datas.eleAt(eleIndex)
                if ele is None:
                    break
                count = count + len(ele['data'][0])  # 累加数据点数
                eleIndex = eleIndex + 1
                need_data.append(ele['data'])  # 收集数据
            return count >= read_point_count, need_data  # 返回是否有足够的数据点和收集的数据

        def read_eeg_data(self, start_millis_second, read_millisecond, freq=250):
            """
            读取指定时间段的脑电图数据，根据模式选择对应的读取方法
            :param start_millis_second: 起始时间戳（毫秒）
            :param read_millisecond: 要读取的时间长度（毫秒），注意不是结束时间戳！
            :param freq: 采样频率（Hz），默认250Hz
            :return: 指定时间段的脑电图数据
            """
            if self.mode == 'serial':
                return self.__read_eeg_from_serial(start_millis_second, read_millisecond, freq)
            elif self.mode == 'tcp':
                return self.__read_eeg_date_tcp(start_millis_second, read_millisecond, freq)

        def __read_eeg_from_serial(self, start_millis_second, read_millisecond, freq=1000):
            """
            串口模式下读取指定时间段的脑电图数据
            :param start_millis_second: 起始时间戳（毫秒）
            :param read_millisecond: 要读取的时间长度（毫秒）
            :param freq: 采样频率（Hz），默认1000Hz
            :return: 指定时间段的脑电图数据，如果数据不足则返回None
            """
            packet_size = freq / self.package_per_second  # 每个数据包的点数
            packet_time = 200  # 每个数据包的时间长度（毫秒）
            point_per_millis = packet_size / packet_time  # 每毫秒的数据点数
            # 以防万一，多取两个包，保证截取时足够长
            packet_num = math.ceil(read_millisecond * point_per_millis / packet_size) + 2
            eeg_data = None
            while self.eeg_datas.length() > 0:
                eeg_left = self.eeg_datas.eleAt(0)  # 获取链表中第一个数据包
                if eeg_left is None:
                    break
                eeg_packet_start_millis = eeg_left['timestamp']  # 数据包的起始时间戳
                # 如果数据包结束时间早于请求的起始时间，则移除该数据包
                if eeg_packet_start_millis + packet_time < start_millis_second:
                    self.eeg_datas.removeHead()
                    continue
                # 确保有足够的数据包
                if self.eeg_datas.length() > packet_num:
                    # 计算数据起始位置
                    eeg_start_position = int((start_millis_second - eeg_packet_start_millis) * point_per_millis)
                    if eeg_start_position < 0:
                        print("eeg time error:{0}".format(eeg_start_position))
                        eeg_start_position = 0
                    need_points = int(read_millisecond * point_per_millis)  # 需要的数据点数
                    eeg_tmp = []
                    # 收集足够的数据包
                    for i in range(packet_num):
                        chan = self.eeg_datas.eleAt(i)
                        eeg_tmp.append(chan['data'])
                    # 将多个通道的数据水平拼接
                    eeg_data = np.hstack(eeg_tmp)
                    # 截取指定时间段的数据
                    eeg_data = eeg_data[:, eeg_start_position:(eeg_start_position + need_points)]
                    break
            return eeg_data

        def devices_received(self, devices):
            """
            接收到设备列表的回调函数
            :param devices: 设备列表
            """
            print("devices count:{0}".format(len(devices)))
            for device in devices:
                print("name:{0},mac:{1},rssi:{2}".format(device['name'], device['mac'], device['rssi']))

        def cmd_error(self, cmd, pm, code):
            """
            命令错误的回调函数
            :param cmd: 命令
            :param pm: 参数
            :param code: 错误代码
            """
            print("cmd error:{0},pm:{1},code:{2}".format(cmd, pm, code))

        def crc_error(self):
            """
            CRC校验错误的回调函数
            """
            print("crc error")

        def array_shape(self, arr):
            """
            递归获取数组的形状
            :param arr: 数组
            :return: 数组的形状
            """
            if isinstance(arr, list):
                return [len(arr)] + self.array_shape(arr[0])
            else:
                return []
