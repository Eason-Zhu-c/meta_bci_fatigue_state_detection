import torch
from scipy.io import loadmat
import numpy as np
from fatigue_detection.model_utils.myNet import SFT_Net
from fatigue_detection.model_utils.toolbox import accuracy_score, label_2class, myDataset_5cv
from fatigue_detection.model_utils.DE_3D_Feature import decompose_to_DE


def process_data(data):
    DE_3D_feature_data = decompose_to_DE(data)
    data = DE_3D_feature_data  # (time_points, channels, freq_bands)

    img_rows, img_cols, num_chan = 6, 9, 5
    data_shape_one = data.shape[0]
    data_4d = np.zeros((data_shape_one, img_rows, img_cols, num_chan))

    # 通道映射到空间位置
    channels = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8', 'CP1', 'CP2',
                'P1', 'PZ', 'P2', 'PO3', 'POZ', 'PO4', 'O1', 'OZ', 'O2']

    # 2D map for 17 channels
    data_4d[:, 0, 0, :] = data[:, 0, :]
    data_4d[:, 0, 8, :] = data[:, 1, :]
    data_4d[:, 1, 0, :] = data[:, 2, :]
    data_4d[:, 1, 8, :] = data[:, 3, :]
    data_4d[:, 2, 0, :] = data[:, 4, :]
    data_4d[:, 2, 8, :] = data[:, 5, :]
    data_4d[:, 2, 3, :] = data[:, 6, :]
    data_4d[:, 2, 5, :] = data[:, 7, :]
    data_4d[:, 3, 3, :] = data[:, 8, :]
    data_4d[:, 3, 4, :] = data[:, 9, :]
    data_4d[:, 3, 5, :] = data[:, 10, :]
    data_4d[:, 4, 3, :] = data[:, 11, :]
    data_4d[:, 4, 4, :] = data[:, 12, :]
    data_4d[:, 4, 5, :] = data[:, 13, :]
    data_4d[:, 5, 3, :] = data[:, 14, :]
    data_4d[:, 5, 4, :] = data[:, 15, :]
    data_4d[:, 5, 5, :] = data[:, 16, :]

    # Reshape to [batch_size, 16, 6, 9, 5]
    data_shape_one //= 16
    data_4d_reshape = np.zeros((data_shape_one, 16, 6, 9, 5))
    for i in range(data_shape_one):
        for j in range(16):
            data_4d_reshape[i, j, :, :, :] = data_4d[i * 16 + j, :, :, :]

    data_4d_reshape = np.swapaxes(data_4d_reshape, 2, 4)
    data_4d_reshape = np.swapaxes(data_4d_reshape, 3, 4)
    return data_4d_reshape


def train_model(data_4d_reshape, file_name, label_file_name, epochs, log_callback=None):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    acc_low = 0.0
    myModel = SFT_Net().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(myModel.parameters(), lr=2e-3, weight_decay=0.02)

    epoch = epochs
    total_train_step = 0
    total_test_step = 0

    label = loadmat(label_file_name)['perclos'][0]
    # la = label_2class(label)

    data_tensor = torch.FloatTensor(data_4d_reshape)
    label_tensor = torch.FloatTensor(label)
    # print(label_tensor)

    train_dataloader, test_dataloader = myDataset_5cv(data_tensor, label_tensor, 150, 0, 20)

    best_acc = 0.0
    best_epoch = -1
    best_model_path = ''

    for i in range(epoch):
        msg = f"--------------The {i + 1}th epoch of training starts------------"
        print(msg)
        if log_callback:
            log_callback(msg)

        total_train_loss = 0
        total_train_acc = 0

        # Training
        myModel.train()
        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)
            outputs, _sa, _fa = myModel(x)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            label_train = y
            label_train_pred = label_2class(outputs)
            total_train_acc += accuracy_score(label_train, label_train_pred)
            total_train_step += 1

        avg_train_loss = total_train_loss / total_train_step
        avg_train_acc = total_train_acc / len(train_dataloader)
        msg = f"train average loss: {avg_train_loss:.6f}\ntrain average accuracy: {100.0 * avg_train_acc:.4f}%"
        print(msg)
        if log_callback:
            log_callback(msg)

        # Testing
        myModel.eval()
        total_test_loss = 0
        total_test_acc = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                outputs, _, _ = myModel(x)
                loss = loss_fn(outputs, y)
                total_test_loss += loss.item()

                label_true = y
                label_pred = label_2class(outputs)
                total_test_acc += accuracy_score(label_true, label_pred)
                total_test_step += 1

        avg_test_loss = total_test_loss / total_test_step
        avg_test_acc = total_test_acc / len(test_dataloader)
        msg = f"test average loss: {avg_test_loss:.6f}\ntest average accuracy: {100.0 * avg_test_acc:.4f}%\n"
        print(msg)
        if log_callback:
            log_callback(msg)

        if avg_test_acc > acc_low:
            acc_low = avg_test_acc
            best_acc = avg_test_acc
            best_epoch = i + 1
            best_model_path = f'model_train/pth/model_fold_{file_name}.pth'
            torch.save(myModel.state_dict(), best_model_path)

    print("-------------------------------------------------------------")
    print(f"Highest accuracy is: {acc_low*100:.4f}% at epoch {best_epoch}")
    # result_text = (
    #     f"The model training is completed.\n"
    #     f"Best epoch: {best_epoch}\n"
    #     f"Best accuracy: {best_acc * 100:.4f}%\n"
    #     f"Model saved at: {best_model_path}"
    # )
    result_text = (
        f"疲劳状态检测模型训练完成.\n"
        f"最佳训练次数: {best_epoch}\n"
        f"最好准确率: {best_acc * 100:.4f}%\n"
        f"模型保存路径: {best_model_path}"
    )

    if log_callback:
        log_callback(result_text)
    return result_text
