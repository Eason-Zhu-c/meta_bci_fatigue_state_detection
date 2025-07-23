import os
from psychopy import visual, event, core
from psychopy.visual import ButtonStim
from .model_viewer import ModelViewerUI
from datetime import datetime
import time
import warnings
import threading
import queue

from metabci.brainflow.amplifiers import NdDevice

import random
import numpy as np
from scipy.io import loadmat
from .Online_fatigue_core import FeedbackWorker
warnings.filterwarnings("ignore")


class Online_fatigueUI:
    def __init__(self, win):
        self.win = win
        self.return_to_main = False
        self.selected_model_file = ''
        self.fatigue_text = None
        self.result_text = None
        self.return_button = None
        self.mouse = event.Mouse(win=self.win)
        self.state_icon = None
        # 获取 fatigue_detection 目录（即上一级目录）
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 当前文件的上级目录
        self.icon_path = os.path.join(current_dir, 'Icons')  # Icons 文件夹应位于 fatigue_detection 目录下
        self.icons = {
            "清醒": "awake.png",
            "疲劳": "fatigue.png",
            "昏睡": "drowsy.png"
        }
        self.i = 0
        self.init_ui()
        self.prediction_queue = queue.Queue()
        self.mode = 'idle'  # 空闲状态（初始界面）
        self.state_icon = visual.ImageStim(
            win=self.win,
            image=None,
            pos=(0, 0.4),
            units='norm',
            size=(0.2, 0.32)
        )
        self.should_stop = False

    def init_ui(self):
        self.title_text = visual.TextStim(self.win, text="实时疲劳状态检测", pos=(0, 0.7),
                                          color="white", height=0.1, font='Microsoft YaHei')
        # 新增：显示选中模型文件名的 TextStim
        self.selected_file_text = visual.TextStim(
            self.win, text="", pos=(0, 0.5), color="yellow", height=0.06, alignText='center', font='Microsoft YaHei'
        )
        self.select_button = ButtonStim(self.win, text="选择已训练好的模型", pos=(0, 0.3), size=(0.5, 0.1),
                                        font='Microsoft YaHei')
        self.connected_button = ButtonStim(self.win, text="开始检测", pos=(0, -0.3), size=(0.5, 0.1),
                                           font='Microsoft YaHei')
        self.back_button = ButtonStim(self.win, text="主页", pos=(-0.7, 0.7), size=(0.25, 0.08), font='Microsoft YaHei')

    def show(self):
        mouse = event.Mouse(win=self.win)
        self.data_list = np.empty((8, 0))

        while True:
            if self.mode == 'idle':
                self.check_prediction_result()
                self.title_text.draw()
                self.select_button.draw()
                self.connected_button.draw()
                self.back_button.draw()

                # 新增：如果已选中模型文件，则显示提示文本
                if self.selected_model_file:
                    file_name = os.path.basename(self.selected_model_file)
                    self.selected_file_text.setText(f"已选择文件: {file_name}")
                    self.selected_file_text.draw()

                self.win.flip()
            elif self.mode == 'predicting':
                self.check_prediction_result()  # 绘制预测界面（疲劳文本 + 图标等）
            # 鼠标点击事件
            if mouse.isPressedIn(self.select_button):
                self.select_model_with_viewer()  # 使用 ModelViewerUI 选择模型
                # core.wait(0.2)
                time.sleep(0.2)
            elif mouse.isPressedIn(self.connected_button) and self.selected_model_file:
                self.mode = 'predicting'  # 切换状态
                collect_thread = threading.Thread(target=self.collect_data)
                predict_thread = threading.Thread(target=self.online_predict)

                collect_thread.start()
                predict_thread.start()

                # self.w = FeedbackWorker(timeout, feedback_worker_name, lsl_source_id, self.selected_model_file,
                #                         self.win, pick_chs, stim_interval, stim_labels, srate)
                # self.w.register_callback(self.update_display)  # 注册回调函数

                # # 创建一个新的数据数组，初始化为 0
                # filtered_data = np.zeros_like(data)
            #
            #     # 定义需要提取的通道的序号
            #     selected_indices = [8, 10, 15, 16]
            #
            #     # 将原数据中对应通道的值复制到新的数据数组中
            #     for idx in selected_indices:
            #         filtered_data[:, idx] = data[:, idx]
            #
            #     data = filtered_data
            #
            #     lsl_source_id = 'meta_online_worker'
            #     feedback_worker_name = 'feedback_worker'
            #     timeout = 0
            #     pick_chs = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4',
            #                 'O1', 'OZ', 'O2']
            #     stim_interval = [0, 8]
            #     stim_labels = ''
            #     srate = 200
            #     w = FeedbackWorker(lsl_source_id, timeout, feedback_worker_name, self.selected_model_file, self.win,
            #                        pick_chs, stim_interval, stim_labels, srate)
            #     w.register_callback(self.update_display)  # 注册回调函数
            #     if w.pre():
            #         w.consume(data)
            #     else:
            #         print(">>> device not connected")
            #     core.wait(0.2)

            elif self.mode == 'idle' and mouse.isPressedIn(self.back_button):
                self.return_to_main = True
                break
            if self.mode == 'predicting' and self.return_button and self.mouse.isPressedIn(self.return_button):
                self.should_stop = True
                self.mode = 'idle'
                self.i = -1
            keys = event.getKeys()
            if 'escape' in keys:
                self.return_to_main = True
                break

        return self.return_to_main

    def select_model_with_viewer(self):
        """使用 ModelViewerUI 来选择模型文件"""
        viewer = ModelViewerUI(self.win)
        viewer.show()  # 显示模型文件列表界面

        if viewer.selected_model_file:
            self.selected_model_file = viewer.selected_model_file
            print(f">>> Selected model file: {self.selected_model_file}")
        else:
            print(">>> No model file selected.")

    def update_display(self, fatigue_state):
        self.prediction_queue.put(fatigue_state)

    # def update_display(self, fatigue_state):
    #     """回调函数：用于动态更新疲劳状态"""
    #     states = {0: "清醒", 1: "疲劳", 2: "昏睡"}
    #     state_str = states.get(fatigue_state, "Unknown")
    #     # 获取当前时间并格式化为字符串
    #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #
    #     if self.fatigue_text is None:
    #         self.fatigue_text = visual.TextStim(
    #             self.win, text="正在实时检测状态...", color="yellow", pos=(0, 0.7),
    #             height=0.08, font='Microsoft YaHei'
    #         )
    #     if self.result_text is None:
    #         self.result_text = visual.TextStim(
    #             self.win, text="", color="white", pos=(0, 0), height=0.08, font='Microsoft YaHei'
    #         )
    #     if self.return_button is None:
    #         self.return_button = ButtonStim(self.win, text="完成", pos=(0, -0.3), size=(0.3, 0.1),
    #                                         font='Microsoft YaHei')
    #     if fatigue_state != '':
    #         self.state_icon = visual.ImageStim(win=self.win, image=os.path.join(self.icon_path, self.icons[state_str]),
    #                                            pos=(0, 0.4), units='norm', size=(0.2, 0.32))
    #     self.fatigue_text.draw()
    #     self.result_text.setText(f"{current_time} - 状态：{state_str}")
    #     self.result_text.draw()
    #     self.return_button.draw()
    #     if self.state_icon:
    #         self.state_icon.draw()
    #     self.win.flip()
    #
    #     # 检测是否点击了 Finish 按钮
    #     should_stop = False
    #     if self.mouse.isPressedIn(self.return_button):
    #         ending_text = visual.TextStim(
    #             self.win, text="在线检测结束！\n返回主界面...",
    #             color="green", height=0.08, pos=(0, 0.3), font='Microsoft YaHei'
    #         )
    #         ending_text.draw()
    #         self.win.flip()
    #         core.wait(1.5)  # 显示提示 1.5 秒
    #         self.return_to_main = True
    #         should_stop = True
    #
    #     return should_stop  # 返回是否需要停止

    def check_prediction_result(self):
        try:
            fatigue_state = self.prediction_queue.get_nowait()
        except queue.Empty:
            return  # 没有新数据就返回

        # 显示状态
        states = {0: "清醒", 1: "疲劳", 2: "昏睡"}
        state_str = states.get(fatigue_state, "Unknown")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.fatigue_text is None:
            self.fatigue_text = visual.TextStim(
                self.win, text="正在实时检测状态...", color="yellow", pos=(0, 0.7),
                height=0.08, font='Microsoft YaHei'
            )
        if self.result_text is None:
            self.result_text = visual.TextStim(
                self.win, text="", color="white", pos=(0, 0), height=0.08, font='Microsoft YaHei'
            )
        if self.return_button is None:
            self.return_button = ButtonStim(self.win, text="完成", pos=(0, -0.3), size=(0.3, 0.1),
                                            font='Microsoft YaHei')
        if fatigue_state != '':
            self.state_icon = visual.ImageStim(win=self.win, image=os.path.join(self.icon_path, self.icons[state_str]),
                                               pos=(0, 0.4), units='norm', size=(0.2, 0.32))
        self.fatigue_text.draw()
        self.result_text.setText(f"{current_time} - 状态：{state_str}")
        self.result_text.draw()
        self.return_button.draw()
        if self.state_icon:
            self.state_icon.draw()
        self.win.flip()

        # 检查是否点了 Finish
        if self.mouse.isPressedIn(self.return_button):
            print("返回")
            # 设置状态，退出预测模式
            self.should_stop = True  # 通知所有线程退出
            self.return_to_main = True
            self.mode = 'idle'
            self.i = -1

            # 清理当前界面元素
            self.fatigue_text = None
            self.result_text = None
            self.return_button = None
            self.state_icon = None

            # 显示“结束检测”提示文字
            ending_text = visual.TextStim(
                self.win, text="在线检测结束！\n返回主界面...",
                color="green", height=0.08, pos=(0, 0.3), font='Microsoft YaHei'
            )
            ending_text.draw()
            self.win.flip()
            time.sleep(1.5)
            return

    def collect_data(self):
        self.nd_device = NdDevice(mode='tcp', com='', tcp_ip='192.168.0.111', tcp_port=8899, host_mac_bytes=None)
        self.nd_device.start()
        while True:
            time.sleep(0.1)
            read_data = self.nd_device.read_latest_eeg_data()  # 读取数据
            # 处理数据格式 - 从(8, N, 1)格式转换为(8, N)
            if read_data is not None and len(read_data.shape) == 3 and read_data.shape[2] == 1:
                read_data = read_data.reshape(read_data.shape[0], read_data.shape[1])
            # 如果需要打印数据，可以取消下面的注释
            if read_data is not None:
                # print(read_data.shape)
                self.data_list = np.hstack((self.data_list, read_data))
            # else:
                # print("No data")

    def online_predict(self):
        # while self.i != -1:
        while not self.should_stop and self.i != -1:
            if self.data_list.shape[1] >= 1600+self.i*200:
                print("training.......................................")
                data = self.data_list[:, 0+self.i*200:1600+self.i*200]
                data = data.T
                data = data[:, 4:]
                data2 = np.zeros((1600, 17))
                data2[:, [8, 10, 14, 16]] = data[:, [0, 1, 2, 3]]

                lsl_source_id = 'meta_online_worker'
                feedback_worker_name = 'feedback_worker'
                timeout = 0
                pick_chs = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4',
                            'O1', 'OZ', 'O2']
                stim_interval = [0, 8]
                stim_labels = ''
                srate = 200
                w = FeedbackWorker(lsl_source_id, timeout, feedback_worker_name, self.selected_model_file, self.win,
                                    pick_chs, stim_interval, stim_labels, srate)
                w.register_callback(self.update_display)  # 注册回调函数
                if w.pre():
                    w.consume(data2)
                self.i += 1