from psychopy import visual, event, core
from psychopy.visual import ButtonStim
from tkinter import Tk, filedialog
import os
from .trainer import process_data, train_model  # 假设 trainer.py 在同级目录下
from .FatigueDataset import *
from .FatigueParadigm import *

class ModelTrainingUI:
    def __init__(self, win):
        self.win = win
        self.return_to_main = False
        self.selected_eeg_file = None
        self.short_name = ""
        self.selected_label_file = None
        self.label_short_name = ""

        self.init_ui()

    def init_ui(self):
        """初始化所有 UI 组件"""
        # 标题文本
        self.title_text = visual.TextStim(
            self.win,
            text="模型训练",
            pos=(0, 0.7),
            color="white",
            height=0.1,
            font='Microsoft YaHei'
        )
        # 新增日志显示区域
        self.log_text = visual.TextStim(
            self.win,
            text="",
            pos=(0, -0.5),
            color="lightgray",
            height=0.04,
            wrapWidth=1.8,
            font='Microsoft YaHei'
        )

        # 选择 EEG 文件按钮
        self.select_button = ButtonStim(self.win, text="选择EEG文件", pos=(0, 0.3), size=(0.5, 0.1), font='Microsoft YaHei')
        # 选择标签文件按钮
        self.select_label_button = ButtonStim(self.win, text="选择标签文件", pos=(0, 0), size=(0.5, 0.1), font='Microsoft YaHei')
        # 开始训练按钮
        self.train_button = ButtonStim(self.win, text="开始训练模型", pos=(0, -0.6), size=(0.5, 0.1), font='Microsoft YaHei')
        # 返回主界面按钮
        self.back_button = ButtonStim(self.win, text="主页", pos=(-0.7, 0.7), size=(0.25, 0.08), font='Microsoft YaHei')

        # 显示文件路径的文本框
        self.file_path_text = visual.TextStim(self.win, text="未选择EEG文件",
                                              pos=(0.6, 0.3), color="white", height=0.04, font='Microsoft YaHei')
        # 显示标签文件路径的文本框
        self.label_path_text = visual.TextStim(self.win, text="未选择标签文件",
                                               pos=(0.6, 0), color="white", height=0.04, font='Microsoft YaHei')

        # 创建输入epoch轮数的框
        self.display = visual.TextStim(self.win, text="请输入训练模型的次数：",
                                       pos=(-0.05, -0.3), color="white",
                                       height=0.04, font='Microsoft YaHei')
        self.input_epochs = visual.TextBox2(self.win, text='', font='Arial', pos=(0.2, -0.3), letterHeight=0.04,
                                            size=(0.1, 0.1), color='white',
                                            borderColor='white')
        self.input = ''

    def show(self):
        """显示界面并处理交互事件"""
        mouse = event.Mouse(win=self.win)
        last_mouse_state = [False, False, False]  # 左、中、右键状态

        while True:
            # 绘制所有组件
            self.title_text.draw()
            self.select_button.draw()
            self.file_path_text.draw()
            self.input_epochs.setText(self.input)
            self.input_epochs.draw()
            self.display.draw()
            self.train_button.draw()
            self.back_button.draw()
            self.select_label_button.draw()
            self.label_path_text.draw()
            self.win.flip()

            # 获取当前鼠标按键状态
            buttons = mouse.getPressed()
            mouse_pos = mouse.getPos()

            keys = event.getKeys()
            for key in keys:
                # 如果按下的键是 backspace，删除最后一个字符
                if key == 'backspace':
                    self.input = self.input[:-1]
                # 如果按下的是回车键且当前输入不为空，保存并返回 True
                elif key == 'return' and self.input.strip():
                    self.epochs = self.input
                    return True
                # 如果按下的是数字键，允许输入
                elif key.isdigit():
                    self.input += key
                # 如果按下的是 escape 键，退出输入
                elif key == 'escape':
                    self.return_to_main = True
                    return False

            # 只有在左键刚按下的那一帧才触发按钮点击逻辑
            if buttons[0] and not last_mouse_state[0]:
                if self.select_button.contains(mouse_pos):
                    self.eeg_file_path = self.select_eeg_file()
                elif self.select_label_button.contains(mouse_pos):
                    self.label_file_path = self.select_label_file()
                elif self.train_button.contains(mouse_pos) and self.selected_eeg_file and self.selected_label_file and self.input.strip():
                    self.epochs = int(self.input)
                    self.start_training(self.epochs)
                    self.selected_eeg_file = None
                    self.selected_label_file = None
                    self.input = ''
                    self.file_path_text.setText("未选择EEG文件")
                    self.label_path_text.setText("未选择标签文件")
                    self.input_epochs.setText("")
                elif self.back_button.contains(mouse_pos):
                    self.return_to_main = True
                    break

            last_mouse_state = buttons[:]  # 更新鼠标状态

            # 检查键盘 ESC 键退出
            keys = event.getKeys()
            if 'escape' in keys:
                self.return_to_main = True
                break

        return self.return_to_main

    def select_eeg_file(self):
        """弹出文件选择对话框"""
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="选择EEG文件",
            filetypes=[
                ("EEG Files", "*.fif *.edf *.bdf *.set *.vhdr *.mat *.csv"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()

        if file_path:
            self.selected_eeg_file = file_path
            # 获取文件名和上级目录名
            dir_name = os.path.basename(os.path.dirname(file_path))  # 上一级目录名
            file_name = os.path.basename(file_path)  # 文件名
            self.short_name = f"{dir_name}/{file_name}"  # 拼接显示名

            # self.short_name = os.path.basename(file_path)
            self.file_path_text.setText(self.short_name)
            print(f">>> Selected file: {self.short_name}")
            return file_path
        else:
            print(">>> No files were selected")

    def select_label_file(self):
        """弹出文件选择对话框，选择标签文件"""
        root = Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="选择标签文件",
            filetypes=[
                ("Label Files", "*.csv *.txt *.mat"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()

        if file_path:
            self.selected_label_file = file_path
            # self.label_short_name = os.path.basename(file_path)
            dir_name = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            self.label_short_name = f"{dir_name}/{file_name}"
            self.label_path_text.setText(self.label_short_name)
            print(f">>> Selected label file: {self.label_short_name}")
            return file_path
        else:
            print(">>> No label files were selected")

    def data_hook(self, X, y, meta, caches):
        X = X.squeeze()
        X = X[:-1, :]
        X = X.T
        X = process_data(X)
        return X, y, meta, caches

    def start_training(self, epochs):
        """触发训练流程"""
        print(">>> The model training is beginning....")
        training_text = visual.TextStim(
            self.win,
            text="正在训练模型！ \n请稍等片刻...",
            color="yellow",
            pos=(0, -0.7), height=0.08, font='Microsoft YaHei')
        training_text.draw()
        self.win.flip()

        # 清空旧日志
        self.log_text.setText("")
        # try:
        dataset = FatigueDataset(self.eeg_file_path)
        paradigm = Fatigue()
        assert paradigm.is_valid(dataset)
        paradigm.register_data_hook(self.data_hook)
        X, _, meta = paradigm.get_data(dataset, subjects=[0], n_jobs=1)
        X = X['whole_segment']
        filename = os.path.basename(self.short_name)
        result_text = train_model(X, filename, self.selected_label_file, epochs, log_callback=self.update_log)
        # except Exception as e:
        #     result_text = f"在训练期间发生错误:\n{str(e)}"
        #     self.update_log(result_text)

        # 最终结果展示
        finish_text = visual.TextStim(
            self.win,
            text=result_text,
            pos=(0, 0.1),
            color="yellow",
            height=0.06,
            font='Microsoft YaHei'
        )
        finish_text.draw()
        self.win.flip()
        core.wait(10)

    def update_log(self, message):
        """用于接收训练过程中的日志信息，并更新到界面上"""
        self.log_text.setText(self.log_text.text + "\n" + message)
        self.log_text.draw()
        self.win.flip()