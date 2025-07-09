import numpy as np
from metabci.brainstim.framework import Experiment
from metabci.brainstim.paradigm import paradigm


class FatigueDetectionParadigm:
    def __init__(self, win):
        self.win = win
        self.name = "Fatigue Detection Paradigm"

    def register_to_experiment(self, ex: Experiment):
        bg_color = np.array([0, 0, 0])
        display_time = 1
        index_time = 1
        rest_time = 0.5
        response_time = 2
        port_addr = None
        nrep = 1
        lsl_source_id = None
        online = False

        def run_fatigue_task(win, expctrl):
            """模拟疲劳检测任务"""
            print("Running fatigue detection task logic...")

        ex.register_paradigm(
            name="Fatigue Detection",
            paradigm_func=paradigm,
            VSObject=self,
            bg_color=bg_color,
            display_time=display_time,
            index_time=index_time,
            rest_time=rest_time,
            response_time=response_time,
            port_addr=port_addr,
            nrep=nrep,
            pdim="fatigue",
            lsl_source_id=lsl_source_id,
            online=online,
            func=run_fatigue_task
        )
        print("Fatigue detection paradigm registered (not running).")
