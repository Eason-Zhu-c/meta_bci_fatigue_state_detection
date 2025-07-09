import os

from metabci.brainstim.framework import Experiment
from psychopy import event, core
from psychopy.visual import ButtonStim, TextStim, ImageStim
from psychopy import monitors
import numpy as np
from data_collection.data_collection_gui import DataCollectionUI
from model_train.model_interface_gui import ModelTrainingUI
from Online_fatigue_detection.Online_fatigue_gui import Online_fatigueUI


class MainExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # main interface
    def main_interface(self):

        win = self.get_window()
        # Unified unit: norm
        win.units = "norm"
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, 'Icons', 'wuyiuniversity.png')

        # 创建图标 stimulus
        icon = ImageStim(
            win=win,
            image=icon_path,
            size=(0.25, 0.4),  # 可调整大小
            pos=(0.87, 0.79)  # (x、y)右上角位置（相对于 norm 单位）
        )

        title_text = TextStim(win, text="基于MetaBCI平台的脑机接口\n疲劳状态检测系统", pos=(0, 0.7),
                              color="white", height=0.1, font='Microsoft YaHei')

        # Create button
        button_collect = ButtonStim(win, text='数据采集', pos=(0, 0.3), size=(0.5, 0.15), font='Microsoft YaHei')
        button_train = ButtonStim(win, text='模型训练', pos=(0, 0), size=(0.5, 0.15), font='Microsoft YaHei')
        button_start = ButtonStim(win, text='疲劳检测', pos=(0, -0.3), size=(0.5, 0.15), font='Microsoft YaHei')

        # Display the buttons and listen for events
        while True:
            icon.draw()
            title_text.draw()
            button_collect.draw()
            button_train.draw()
            button_start.draw()
            win.flip()  # Update the window content

            mouse = event.Mouse()
            if mouse.getPressed()[0]:
                if button_collect.contains(mouse):
                    self.collect_data()
                elif button_train.contains(mouse):
                    self.train_model()
                elif button_start.contains(mouse):
                    self.Online_fatigue_detection()

            if "escape" in event.getKeys():
                core.quit()

    def Online_fatigue_detection(self):
        print(">>> The Initiation Paradigm button was clicked")
        win = self.get_window()
        online_ui = Online_fatigueUI(win)

        should_return = online_ui.show()  # Display the gui and wait for the result

        if not should_return:
            print(">>> continue data collection...")
        else:
            print(">>> Return to the main interface")

    # The functional function corresponding to the button
    def collect_data(self):
        print(">>> The data collection button was clicked")
        win = self.get_window()
        dc_ui = DataCollectionUI(win)

        should_return = dc_ui.show()  # Display the gui and wait for the result

        if not should_return:
            print(">>> continue data collection...")
        else:
            print(">>> Return to the main interface")

    def train_model(self):
        print(">>> The model train button is clicked")
        win = self.get_window()
        mt_ui = ModelTrainingUI(win)
        should_return = mt_ui.show()  # Display the gui and wait for the result

        if not should_return:
            print(">>> model training process...")
        else:
            print(">>> Return to the main interface")


# 入口函数
def main():
    mon = monitors.Monitor(name='primary_monitor', width=59.6, distance=60)
    mon.setSizePix([1920, 1080])

    ex = MainExperiment(
        monitor=mon,
        bg_color_warm=np.array([0, 0, 0]),
        screen_id=0,
        win_size=[1280, 800],
        # win_size=[1920, 1080],
        is_fullscr=False,
        record_frames=False,
        disable_gc=False,
        process_priority='normal',
        use_fbo=False)

    ex.main_interface()


if __name__ == '__main__':
    main()
