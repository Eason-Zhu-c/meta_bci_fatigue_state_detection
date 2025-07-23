import numpy as np
from scipy.io import loadmat
from mne import create_info
from mne.io import RawArray

# === Step 1: 加载 EEG 数据 ===
def setup_event(data_path):
    eeg_mat = loadmat(data_path, squeeze_me=True)
    X_raw = eeg_mat['EEG']['data']  # shape: (n_samples, n_channels)
    X_raw = X_raw.item()

    # 创建一个新的数据数组，初始化为 0
    filtered_data = np.zeros_like(X_raw)

    # 定义需要提取的通道的序号
    selected_indices = [8, 10, 15, 16]

    # 将原数据中对应通道的值复制到新的数据数组中
    for idx in selected_indices:
        filtered_data[:, idx] = X_raw[:, idx]

    X_raw = filtered_data

    # 若为 memoryview/object，则解包
    if X_raw.ndim == 1 and isinstance(X_raw[0], (np.ndarray, list)):
        X_raw = np.stack(X_raw)

    # shape: (n_channels, n_samples)
    if X_raw.shape[0] > X_raw.shape[1]:
        X_raw = X_raw.T

    # === 构造 stim 通道 ===
    # n_samples = X_raw.shape[1]
    # stim = np.zeros((1, n_samples))
    stim = np.zeros((1, X_raw.shape[1]))  # 初始化 stim 通道
    stim[0, 0] = 1  # 在第0个采样点打一个 event_id = 1

    # === 合并 stim 通道 ===
    X_all = np.vstack([X_raw, stim])  # (n_channels + 1, n_samples)

    # ch_names = [f'Ch{i+1}' for i in range(17)] + ['STI 014']
    ch_names = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
                'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2', 'STI 014']
    ch_types = ['eeg'] * 17 + ['stim']
    info = create_info(ch_names=ch_names, sfreq=200, ch_types=ch_types)
    raw = RawArray(X_all, info)
    print("X_all shape:", X_all.shape)
    return raw

