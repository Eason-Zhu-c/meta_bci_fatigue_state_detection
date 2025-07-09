# subject_input.py
from psychopy import visual, event, core
from psychopy.visual import ButtonStim


class SubjectNameInputUI:
    def __init__(self, win):
        self.win = win
        self.subject_name = None
        # self.return_to_main = False
        self.mouse = event.Mouse(win=win)

    def show(self):
        event.clearEvents()

        prompt = visual.TextStim(self.win, text="请输入受试者姓名：", pos=(0, 0.3), color="white", height=0.08,
                                 font='Microsoft YaHei')
        input_box = visual.TextBox2(self.win, text='', font='Arial', pos=(0, 0), letterHeight=0.08,
                                    size=(1.0, 0.15), color='white', borderColor='white')
        confirm_button = ButtonStim(self.win, text='确认', pos=(-0.15, -0.3), size=(0.2, 0.1), font='Microsoft YaHei')
        back_button = ButtonStim(self.win, text='返回', pos=(0.15, -0.3), size=(0.2, 0.1), font='Microsoft YaHei')
        current_input = ''

        while True:
            prompt.draw()
            input_box.setText(current_input)
            input_box.draw()
            confirm_button.draw()
            back_button.draw()
            self.win.flip()

            keys = event.getKeys()
            for key in keys:
                if key == 'backspace':
                    current_input = current_input[:-1]
                elif key == 'return' and current_input.strip():
                    self.subject_name = current_input
                    return True
                elif len(key) == 1:
                    current_input += key
                elif key == 'escape':
                    # self.return_to_main = True
                    return False

            if self.mouse.isPressedIn(confirm_button) and current_input.strip():
                self.subject_name = current_input
                return True
            elif self.mouse.isPressedIn(back_button):
                # self.return_to_main = True
                return False
