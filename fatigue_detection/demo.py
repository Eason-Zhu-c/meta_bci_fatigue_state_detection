import math

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    paradigm,
    fatigue,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix

if __name__ == "__main__":
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,
        verbose=False,
    )
    mon.setSizePix([1920, 1080])
    mon.save()

    bg_color_warm = np.array([0, 0, 0])
    # win_size = np.array([1920, 1080])
    win_size = np.array([1280, 800])

    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,
        screen_id=0,
        win_size=win_size,
        is_fullscr=False,
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    # 注册 fatigue 范式
    fatigue = fatigue(win)
    fatigue.monitor = mon
    fatigue.bg_color_warm = bg_color_warm
    fatigue.screen_id = 0
    fatigue.win_size = win_size
    fatigue.is_fullscr = False
    fatigue.record_frames = False
    fatigue.disable_gc = False
    fatigue.process_priority = "normal"
    fatigue.use_fbo = False

    ex.register_paradigm(
        "fatigue",
        paradigm,
        VSObject=fatigue,
        bg_color=np.array([0.3, 0.3, 0.3]),
        display_time=1,
        index_time=1,
        rest_time=0.5,
        response_time=1,
        port_addr=None,
        nrep=1,
        pdim="fatigue",
        lsl_source_id=None,
        online=False,
    )
    """
       SSVEP
       """
    n_elements, rows, columns = 20, 4, 5  # n_elements 指令数量;  rows 行;  columns 列
    stim_length, stim_width = 200, 200  # ssvep单指令的尺寸
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]  # 指令的颜色，文字的颜色
    fps = 240  # 屏幕刷新率
    stim_time = 2  # 刺激时长
    stim_opacities = 1  # 刺激对比度
    freqs = np.arange(8, 16, 0.4)  # 指令的频率
    phases = np.array([i * 0.35 % 2 for i in range(n_elements)])  # 指令的相位

    basic_ssvep = SSVEP(win=win)

    basic_ssvep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssvep.config_text(tex_color=tex_color)
    basic_ssvep.config_color(
        refresh_rate=fps,
        stim_time=stim_time,
        stimtype="sinusoid",
        stim_color=stim_color,
        stim_opacities=stim_opacities,
        freqs=freqs,
        phases=phases,
    )
    basic_ssvep.config_index()
    basic_ssvep.config_response()

    bg_color = np.array([0.3, 0.3, 0.3])  # 背景颜色
    display_time = 1  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    port_addr = "COM8"  # 0xdefc                                  # 采集主机端口
    port_addr = None  # 0xdefc
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSVEP",
        paradigm,
        VSObject=basic_ssvep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssvep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    # 启动实验
    ex.run()
