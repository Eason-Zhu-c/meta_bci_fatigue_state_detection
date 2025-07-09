from datetime import datetime

from psychopy import visual, event, core


class FatiguePromptUI:
    def __init__(self, win):
        self.win = win

    def show_prompt(self):
        prompt_text = visual.TextStim(
            self.win,
            # text="请选择您的当前疲劳状态（小键盘1-4）：\n\n\n1. 清醒\n\n2. 轻度疲劳\n\n3. 中度疲劳\n\n4. 重度疲劳",
            text="请选择您的当前疲劳状态（小键盘1-3）：\n\n\n1. 清醒\n\n2. 疲劳\n\n3. 昏睡",
            color="white", height=0.08, pos=(0, 0.3), font='Microsoft YaHei'
        )
        prompt_text.draw()
        self.win.flip()

        prompt_start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        timer = core.CountdownTimer(5.0)
        while timer.getTime() > 0:
            keys = event.getKeys(keyList=['num_1', 'num_2', 'num_3'], timeStamped=True)
            if keys:
                key, timestamp = keys[0]
                key_pressed = keys[0][0].replace('num_', '')
                key_press_time_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                return key_pressed, prompt_start_time_str, key_press_time_str
            core.wait(0.01)
        # 超时，默认为-1
        key_press_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return -1, prompt_start_time_str, key_press_time_str  # 默认值
