from typing import Dict
from metabci.brainda.datasets.base import BaseDataset  # 引用 BaseDataset
from .setup_event import *
from mne.io import Raw


class FatigueDataset(BaseDataset):
    # 数据集的通道名
    _CHANNELS = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
                 'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2', 'STI 014']

    # 假设这是一系列的事件标签（包括频率和阶段等）
    _EVENTS = {
        "whole_segment": (1, (0, 7080))  # 一个事件，从第0秒起取 70.8 秒（1416000 / 200）
    }

    def __init__(self, X_path):
        self.X_path = X_path
        super().__init__(
            dataset_code="fatigue_data",  # 数据集唯一标识
            subjects=[0],
            events=self._EVENTS,  # 刺激事件标签
            channels=self._CHANNELS,  # EEG 通道名
            srate=200,  # 假设采样率为 200 Hz
            paradigm="fatigue",  # 实验范式
        )

    def data_path(self, subject, path=None, force_update=False, update_path=None, proxies=None, verbose=None):
        return [[self.X_path]]

    def _get_single_subject_data(self, subject, verbose=None) -> Dict[str, Dict[str, Raw]]:
        # 加载单个被试的原始数据
        X_path = self.data_path(0)[0][0]
        raw = setup_event(X_path)
        return {
            'session1': {
                'run1': raw
            }
        }
