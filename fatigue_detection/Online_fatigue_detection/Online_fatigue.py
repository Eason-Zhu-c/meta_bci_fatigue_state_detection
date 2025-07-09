from psychopy import visual, event, core
from psychopy.visual import ButtonStim
from metabci.brainflow.workers import ProcessWorker
from fatigue_detection.model_utils.myNet import *
import torch
from torch.utils.data import Dataset, DataLoader
from fatigue_detection.model_utils.DE_3D_Feature import *
from scipy.io import loadmat
import warnings
from metabci.brainflow.amplifiers import NeuroScan
import time
from .model_viewer import ModelViewerUI
warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def label_2class(a):
    label_2classes = []
    # a = torch.squeeze(a)
    for i in range(0, len(a)):
        if a[i] < .35:
            label_2classes.append(0)
        elif a[i] < .7:
            label_2classes.append(1)
        else:
            label_2classes.append(2)
    return label_2classes

class Online_fatigueUI:
    def __init__(self, win):
        self.win = win
        self.return_to_main = False
        self.selected_model_file = ''
        self.init_ui()

    def init_ui(self):
        self.title_text = visual.TextStim(self.win, text="Online fatigue detection interface", pos=(0, 0.7), color="white", height=0.1)
        self.select_button = ButtonStim(self.win, text="Select the model", pos=(0, 0.3), size=(0.5, 0.1))
        self.connected_button = ButtonStim(self.win, text="Connected Device", pos=(0, -0.3), size=(0.5, 0.1))
        self.back_button = ButtonStim(self.win, text="HomePage", pos=(-0.7, 0.7), size=(0.25, 0.08))

    def show(self):
        mouse = event.Mouse(win=self.win)

        while True:
            self.title_text.draw()
            self.select_button.draw()
            self.connected_button.draw()
            self.back_button.draw()
            self.win.flip()

            # 鼠标点击事件
            if mouse.isPressedIn(self.select_button):
                self.select_model_with_viewer()  # 使用 ModelViewerUI 选择模型
                core.wait(0.2)

            elif mouse.isPressedIn(self.connected_button) and self.selected_model_file:
                data_path = r'E:\朱艺森\实验室\BCI比赛\2025\metaBCI\SEED-VIG\Raw_Data\12_20150928_noon.mat'
                data = loadmat(data_path)['EEG']['data'][0, 0]

                lsl_source_id = 'meta_online_worker'
                feedback_worker_name = 'feedback_worker'
                timeout = 0
                pick_chs = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2', 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']
                stim_interval = [0, 8]
                stim_labels = ''
                srate = 200
                w = FeedbackWorker(lsl_source_id, timeout, feedback_worker_name, self.selected_model_file, self.win, pick_chs, stim_interval, stim_labels, srate)

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

# 自定义测试数据集类（没有标签）
class YourTestDataset(Dataset):
    def __init__(self, data):
        """
        初始化数据集
        :param data: 输入数据，通常是一个NumPy数组或Tensor
        """
        self.data = data  # 输入数据

    def __len__(self):
        """
        返回数据集的样本数
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回指定索引的数据（没有标签）
        :param idx: 索引
        :return: 输入数据
        """
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

class FeedbackWorker(ProcessWorker):
    def __init__(self,
                 timeout, # 超时设置
                 worker_name, # Worker 名称
                 lsl_source_id, # LSL 数据流的源 ID
                 file_path, # 实验数据文件路径
                 win,   #屏幕
                 pick_chs,  # 选择的EEG通道
                 stim_interval,  # 刺激的时间窗口（数据截取区间）
                 stim_labels,  # 事件标签
                 srate,  # 采样率
                 ):
        self.lsl_source_id = lsl_source_id  # LSL 数据流的源 ID
        self.file_path = file_path  # 实验数据文件路径
        self.win = win
        self.pick_chs = pick_chs  # 保存选择的EEG通道
        self.stim_interval = stim_interval  # 刺激时间窗口
        self.stim_labels = stim_labels  # 事件标签
        self.srate = srate  # 采样率
        super().__init__(timeout=timeout, name=worker_name)

    def pre(self):
        """
        预处理函数，在 worker 启动时执行。
        1. 获得训练模型
        2. 设置 LSL 数据流
        """
        self.estimator = SFT_Net()
        self.estimator.load_state_dict(torch.load(self.file_path, map_location=device))
        # 将模型移动到设备（GPU 或 CPU）
        self.estimator.to(device)
        self.estimator.eval()

        ns = NeuroScan(
            device_address=('192.168.56.5', 4000),
            srate=self.srate,
            num_chans=17)  # NeuroScan parameter

        # 与ns建立tcp连接
        # ns.connect_tcp()
        # ns开始采集波形数据
        # ns.start_acq()

        # register worker来实现在线处理
        # ns.register_worker(feedback_worker_name, worker, marker)

        # 开启在线处理进程
        # ns.up_worker(feedback_worker_name)
        # 等待 0.5s
        # time.sleep(0.5)

        # ns开始截取数据线程，并把数据传递数据给处理进程
        # ns.start_trans()

        # 设置 LSL 数据流信息
        # info = StreamInfo(
        #     name='meta_feedback',  # 数据流名称
        #     type='Markers',  # 数据流类型（Markers 类型）
        #     channel_count=1,  # 只有一个通道
        #     nominal_srate=0,  # 数据流的采样率
        #     channel_format='int32',  # 数据流的格式（int32）
        #     source_id=self.lsl_source_id)  # 设置 LSL 数据源 ID
        # self.outlet = StreamOutlet(info)  # 创建 LSL 数据流的输出通道

        # print('Waiting for connection...')  # 等待连接
        # while not self._exit:  # 循环直到连接消费者
        #     if self.outlet.wait_for_consumers(1e-3):  # 等待消费者连接（1 毫秒超时）
        #         break
        # print('Connected')  # 连接成功
        return True

    def consume(self, data):
        """
        数据消费函数：每次采集到新数据时执行。
        1. 对数据进行预处理
        2. 使用训练好的模型进行推理（分类）
        3. 输出分类结果
        """

        #拿到的数据只有两个维度（时间点，通道）
        X = np.empty([0, 17, 5])
        DE_3D_feature_data = decompose_to_DE(data)
        data = np.vstack([X, DE_3D_feature_data])

        data_shape_one = data.shape[0]
        img_rows, img_cols, num_chan = 6, 9, 5
        data_4d = np.zeros((data_shape_one, img_rows, img_cols, num_chan))

        channels = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
                    'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']

        # 2D map for 17 channels
        # 'FT7'(channel1) :
        data_4d[:, 0, 0, :] = data[:, 0, :]
        # 'FT8'(channel2) :
        data_4d[:, 0, 8, :] = data[:, 1, :]
        # 'T7' (channel3) :
        data_4d[:, 1, 0, :] = data[:, 2, :]
        # 'T8' (channel4) :
        data_4d[:, 1, 8, :] = data[:, 3, :]
        # 'TP7'(channel5) :
        data_4d[:, 2, 0, :] = data[:, 4, :]
        # 'TP8'(channel6) :
        data_4d[:, 2, 8, :] = data[:, 5, :]
        # 'CP1'(channel7) :
        data_4d[:, 2, 3, :] = data[:, 6, :]
        # 'CP2'(channel8) :
        data_4d[:, 2, 5, :] = data[:, 7, :]
        # 'P1' (channel9) :
        data_4d[:, 3, 3, :] = data[:, 8, :]
        # 'PZ' (channel10):
        data_4d[:, 3, 4, :] = data[:, 9, :]
        # 'P2' (channel11):
        data_4d[:, 3, 5, :] = data[:, 10, :]
        # 'PO3'(channel12):
        data_4d[:, 4, 3, :] = data[:, 11, :]
        # 'POZ'(channel13):
        data_4d[:, 4, 4, :] = data[:, 12, :]
        # 'PO4'(channel14):
        data_4d[:, 4, 5, :] = data[:, 13, :]
        # 'O1' (channel15):
        data_4d[:, 5, 3, :] = data[:, 14, :]
        # 'OZ' (channel16):
        data_4d[:, 5, 4, :] = data[:, 15, :]
        # 'O2' (channel17):
        data_4d[:, 5, 5, :] = data[:, 16, :]

        # from 3D features to 4D features
        # [data_shape_one, 6, 9, 5] -> [data_shape_one//16, 16, 6, 9, 5]
        data_shape_one //= 16
        data_4d_reshape = np.zeros((data_shape_one, 16, 6, 9, 5))
        for i in range(data_shape_one):
            for j in range(16):
                data_4d_reshape[i, j, :, :, :] = data_4d[i * 16 + j, :, :, :]

        # [data_shape_one//16, 16, 6, 9, 5] -> [data_shape_one//16, 16, 5, 6, 9]
        data_4d_reshape = np.swapaxes(data_4d_reshape, 2, 4)
        data_4d_reshape = np.swapaxes(data_4d_reshape, 3, 4)

        test_dataset = YourTestDataset(data_4d_reshape)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # p_labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs, _, _ = self.estimator(data)
                label_pred = label_2class(outputs)
                label_pred = label_pred[0]
                # p_labels.append(label_pred)
                if label_pred == 0:
                    print("<<< 受试者状态：清醒")
                    self.fatigue = "Awake"
                    self.display_result()
                if label_pred == 1:
                    print("<<< 受试者状态：疲劳")
                    self.fatigue = "Fatigue"
                    self.display_result()
                if label_pred == 2:
                    print("<<< 受试者状态：昏昏欲睡")
                    self.fatigue = "Drowsy"
                    self.display_result()
                core.wait(0.2)
            return

        # 如果有消费者连接（例如显示系统、反馈系统等）
        # if self.outlet.have_consumers():
        #     # 将预测结果通过 LSL 数据流输出
        #     self.outlet.push_sample(p_labels)

    def post(self):
        """
        后处理函数，在任务完成后执行。
        目前没有额外的清理工作，留作以后使用。
        """
        pass

    def display_result(self):
        fatigue_text = visual.TextStim(
            self.win, text="Fatigue detection is in progress", color="yellow", pos=(0, 0.7),
            height=0.08
        )
        result_text = visual.TextStim(
            self.win, text=f"The current state of the subject is {self.fatigue}", color="white", pos=(0, 0),
            height=0.08
        )
        return_button = ButtonStim(self.win, text="Finish", pos=(0, -0.3), size=(0.3, 0.1))

        fatigue_text.draw()  # 绘制训练提示文本
        result_text.draw()
        return_button.draw()

        mouse = event.Mouse(win=self.win)
        if mouse.isPressedIn(return_button):
            # 提示用户数据采集结束，即将返回主界面
            ending_text = visual.TextStim(
                self.win, text="fatigue detection ended.\nReturning to previous menu...",
                color="green", height=0.08, pos=(0, 0.3)
            )
            ending_text.draw()
            self.win.flip()
            core.wait(1.5)  # 显示提示 1.5 秒
            # 任意键关闭处理进程——按下任意键结束
            # input('press any key to close\n')
            # 关闭处理进程——停止worker
            # ns.down_worker('feedback_worker')
            # 等待 1s
            # time.sleep(1)

            # ns停止在线截取线程——停止向worker传数据
            # ns.stop_trans()
            # ns停止采集波形数据——停止采集数据
            # ns.stop_acq()
            # ns.close_connection()  # 与ns断开连接——停止连接
            # ns.clear()  # 放大器关闭
            # print('bye')
            self.return_to_main = True
            return
        self.win.flip()  # 更新窗口


if __name__ == '__main__':
    # 放大器的采样率
    srate = 1000
    # 截取数据的时间段（时间窗口的大小）
    stim_interval = [0, 8]

    pick_chs = ['FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
                'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'P5',
                'P3', 'P1', 'PZ', 'P2', 'P4', 'P6']

    lsl_source_id = 'meta_online_worker'
    feedback_worker_name = 'feedback_worker'

    data_path = r'D:\000project\dataset\SEED-VIG\Raw_Data\1_20151124_noon_2.mat'
    data = loadmat(data_path)['EEG']['data'][0, 0]
    w = FeedbackWorker(lsl_source_id, 0, feedback_worker_name)
    w.pre()
    w.consume(data)

    # 放大器的定义
    ns = NeuroScan(
        device_address=('192.168.56.5', 4000),
        srate=srate,
        num_chans=68)  # NeuroScan parameter

    # 与ns建立tcp连接——建立连接
    ns.connect_tcp()
    # ns开始采集波形数据——数据采集
    ns.start_acq()

    # register worker来实现在线处理——注册worker
    ns.register_worker(feedback_worker_name, worker, marker)
    # 开启在线处理进程——启动worker
    ns.up_worker(feedback_worker_name)
    # 等待 0.5s
    time.sleep(0.5)

    # ns开始截取数据线程，并把数据传递数据给处理进程——向worker传数据
    ns.start_trans()

    # 任意键关闭处理进程——按下任意键结束
    input('press any key to close\n')
    # 关闭处理进程——停止worker
    ns.down_worker('feedback_worker')
    # 等待 1s
    time.sleep(1)

    # ns停止在线截取线程——停止向worker传数据
    ns.stop_trans()
    # ns停止采集波形数据——停止采集数据
    ns.stop_acq()
    ns.close_connection()  # 与ns断开连接——停止连接
    ns.clear()  # 放大器关闭
    print('bye')
