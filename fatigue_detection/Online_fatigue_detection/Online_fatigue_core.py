import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from psychopy import core

from metabci.brainflow.workers import ProcessWorker
from metabci.brainda.algorithms.model_utils.myNet import SFT_Net
from metabci.brainda.algorithms.model_utils.DE_3D_Feature import decompose_to_DE
from metabci.brainflow.amplifiers import NdDevice

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 每秒钟接收的数据包数量
# package_per_second = 5
# # 存储5分钟的数据，计算存储的包数量
# eeg_package_count = package_per_second * 60 * 5  # 脑电图数据包数量
# eog_package_count = package_per_second * 60 * 5  # 眼电图数据包数量


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
                 timeout,  # 超时设置
                 worker_name,  # Worker 名称
                 lsl_source_id,  # LSL 数据流的源 ID
                 file_path,  # 实验数据文件路径
                 win,  #屏幕
                 pick_chs,  # 选择的EEG通道
                 stim_interval,  # 刺激的时间窗口（数据截取区间）
                 stim_labels,  # 事件标签
                 srate,  # 采样率
                 ):
        super().__init__(timeout=timeout, name=worker_name)
        self.lsl_source_id = lsl_source_id  # LSL 数据流的源 ID
        self.file_path = file_path  # model文件路径
        self.win = win
        self.pick_chs = pick_chs  # 保存选择的EEG通道
        self.stim_interval = stim_interval  # 刺激时间窗口
        self.stim_labels = stim_labels  # 事件标签
        self.srate = srate  # 采样率

        self.on_update_callback = None  # 回调函数

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
        return True
        # try:
        #     self.nd_device = NdDevice(eeg_package_count, eog_package_count, mode='tcp', com='',
        #                                    tcp_ip='192.168.0.111', tcp_port=8899, host_mac_bytes=None)
        #     self.nd_device.start()  # 启动设备
        #
        #     index = 0
        #     while index < 100:  # 循环10次
        #         time.sleep(0.1)  # 每0.1秒读取一次数据
        #
        #         index = index + 1
        #         millis_second = int(round(time.time() * 1000))  # 获取当前时间戳（毫秒）
        #         time_span = 1000  # 读取过去1秒的数据
        #         read_data = self.nd_device.read_latest_eeg_data()  # 读取数据
        #         if read_data is not None:
        #             print(read_data.shape)
        #             # nd_device.close()
        #             self.connect = 1
        #             break
        # except:
        #     self.connect = 0
        # print('self.connect', self.connect)
        # return self.connect

        # ns = NeuroScan(
        #     device_address=('192.168.56.5', 4000),
        #     srate=self.srate,
        #     num_chans=17)  # NeuroScan parameter

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
        # return True

    def register_callback(self, callback):
        """注册回调函数"""
        self.on_update_callback = callback

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
        print(X.shape)
        data_shape_one = data.shape[0]
        # print(X_shape_one)
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
                print('label_pred', label_pred)
                if self.on_update_callback:
                    stop_flag = self.on_update_callback(label_pred)  # 接收返回值
                    if stop_flag:  # 如果返回 True，则终止检测
                        print(">>> The user clicks Finish to stop the online detection")
                        return  # 直接返回，不再继续处理
                # core.wait(0.2)  # 模拟在线间隔
                time.sleep(0.2)
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

