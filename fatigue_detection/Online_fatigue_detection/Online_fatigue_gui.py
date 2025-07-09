import os
import random

from psychopy import visual, event, core
from psychopy.visual import ButtonStim
from scipy.io import loadmat
from .Online_fatigue_core import FeedbackWorker
from .model_viewer import ModelViewerUI
from datetime import datetime
import warnings
import numpy as np

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
        self.init_ui()

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
        # 定义 test 文件夹中的 mat 文件列表
        current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前 .py 文件路径
        test_folder = os.path.join(current_dir, 'test')  # test 文件夹路径
        mat_files = {
            "1_": "1_20151124_noon_2.mat",
            "2_": "2_20151106_noon.mat",
            "3_": "3_20151024_noon.mat",
            "4_": "4_20151105_noon.mat"
        }

        while True:
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
            # 鼠标点击事件
            if mouse.isPressedIn(self.select_button):
                self.select_model_with_viewer()  # 使用 ModelViewerUI 选择模型
                core.wait(0.2)

            elif mouse.isPressedIn(self.connected_button) and self.selected_model_file:
                # 根据 selected_model_file 决定加载哪个 mat 文件
                selected_mat = None
                # 检查模型文件名中是否包含某个前缀
                for prefix, filename in mat_files.items():
                    if prefix in self.selected_model_file:
                        selected_mat = filename
                        print(f">>> Find the corresponding file:{selected_mat}, {self.selected_model_file}")
                        break
                # 如果没有匹配的前缀，则随机选一个
                if not selected_mat:
                    selected_mat = os.path.join(test_folder, random.choice(list(mat_files.values())))
                # 构造完整的 data_path
                data_path = os.path.join(test_folder, selected_mat)
                # data_path = r'E:\朱艺森\实验室\BCI比赛\2025\metaBCI\SEED-VIG\Raw_Data\12_20150928_noon.mat'
                data = loadmat(data_path)['EEG']['data'][0, 0]

                # 创建一个新的数据数组，初始化为 0
                filtered_data = np.zeros_like(data)

                # 定义需要提取的通道的序号
                selected_indices = [8, 10, 15, 16]

                # 将原数据中对应通道的值复制到新的数据数组中
                for idx in selected_indices:
                    filtered_data[:, idx] = data[:, idx]

                data = filtered_data

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
                    w.consume(data)
                else:
                    print(">>> device not connected")
                core.wait(0.2)

            elif mouse.isPressedIn(self.back_button):
                self.return_to_main = True
                break

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
        """回调函数：用于动态更新疲劳状态"""
        states = {0: "清醒", 1: "疲劳", 2: "昏睡"}
        state_str = states.get(fatigue_state, "Unknown")
        # 获取当前时间并格式化为字符串
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

        # 检测是否点击了 Finish 按钮
        should_stop = False
        if self.mouse.isPressedIn(self.return_button):
            ending_text = visual.TextStim(
                self.win, text="在线检测结束！\n返回主界面...",
                color="green", height=0.08, pos=(0, 0.3), font='Microsoft YaHei'
            )
            ending_text.draw()
            self.win.flip()
            core.wait(1.5)  # 显示提示 1.5 秒
            self.return_to_main = True
            should_stop = True

        return should_stop  # 返回是否需要停止
