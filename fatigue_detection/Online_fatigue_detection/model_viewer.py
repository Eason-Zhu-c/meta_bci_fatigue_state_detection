from psychopy import visual, event, core
from psychopy.visual import ButtonStim
import os


class ModelViewerUI:
    def __init__(self, win):
        self.win = win
        self.return_to_main = False
        self.folder_path = "model_train"  # 固定目录名，与主程序同级
        self.file_list = []
        self.buttons = []
        self.current_page = 0
        self.files_per_page = 10  # 每页最多显示 10 个文件
        self.selected_model_file = None  # 存储最终选中的模型文件路径
        self.temp_selected_index = None  # 临时存储点击的文件索引

        # UI 组件初始化
        self.title = visual.TextStim(self.win, text="状态监测模型列表", pos=(0, 0.85), color="white", height=0.08, font='Microsoft YaHei')

        # 新增：显示选中文件名的文本组件
        self.selected_file_text = visual.TextStim(self.win, text="", pos=(0, 0.7), color="yellow", height=0.05)

        self.no_file_text = visual.TextStim(self.win, text="未选择模型文件.", pos=(0, 0), color="red",
                                            height=0.08, font='Microsoft YaHei')

        # 分页按钮
        self.prev_button = ButtonStim(self.win, text="上一页", pos=(-0.3, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')
        self.next_button = ButtonStim(self.win, text="下一页", pos=(-0.1, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')

        # 确认按钮
        self.confirm_button = ButtonStim(self.win, text="确认", pos=(0.1, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')
        self.back_button = ButtonStim(self.win, text="返回", pos=(0.3, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')

        self.init_ui()

    def init_ui(self):
        """初始化界面：加载模型文件列表"""
        model_dir = os.path.join(self.folder_path, "pth")  # 指向 model_train/pth/
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            self.file_list = sorted([f for f in os.listdir(model_dir) if f.endswith(".pth")])
        else:
            self.file_list = []

        if self.file_list:
            self.update_buttons_for_current_page()
        else:
            print(">>> No .pth model files found in the directory.")

    def update_buttons_for_current_page(self):
        """根据当前页码更新文件按钮列表"""
        self.buttons.clear()
        y_start = 0.6
        spacing = -0.12

        start_idx = self.current_page * self.files_per_page
        end_idx = min(start_idx + self.files_per_page, len(self.file_list))

        for idx, filename in enumerate(self.file_list[start_idx:end_idx]):
            x_pos = 0
            y_pos = y_start + idx * spacing
            btn = ButtonStim(self.win, text=filename, pos=(x_pos, y_pos), size=(1.0, 0.08))
            self.buttons.append(btn)

    def show(self):
        """主循环：显示模型文件列表并处理交互"""
        mouse = event.Mouse(win=self.win)

        while True:
            # 绘制所有组件
            self.title.draw()
            self.back_button.draw()

            # 新增：绘制选中文件名文本
            self.selected_file_text.draw()

            if not self.file_list:
                self.no_file_text.draw()
            else:
                for btn in self.buttons:
                    btn.draw()
                self.prev_button.draw()
                self.next_button.draw()
                self.confirm_button.draw()

            self.win.flip()

            # 键盘事件
            keys = event.getKeys()
            if 'escape' in keys:
                self.return_to_main = True
                break

            # 鼠标点击事件
            if mouse.isPressedIn(self.back_button):
                self.return_to_main = True
                break

            if self.file_list:
                # 点击文件按钮：仅临时记录索引
                for idx, btn in enumerate(self.buttons):
                    if mouse.isPressedIn(btn):
                        selected_index = self.current_page * self.files_per_page + idx
                        self.temp_selected_index = selected_index
                        selected_file_name = self.file_list[selected_index]
                        self.selected_file_text.setText(f"已选择文件: {selected_file_name}")
                        print(f">>> Temporarily selected file index: {selected_index}")
                        core.wait(0.2)

                # 点击确认按钮：真正确认选择
                if mouse.isPressedIn(self.confirm_button) and self.temp_selected_index is not None:
                    self.selected_model_file = os.path.join(self.folder_path, "pth",
                                                            self.file_list[self.temp_selected_index])
                    print(f">>> Confirmed model file: {self.selected_model_file}")
                    self.return_to_main = True
                    break

            # 上一页
            if mouse.isPressedIn(self.prev_button) and self.current_page > 0:
                self.current_page -= 1
                self.update_buttons_for_current_page()
                core.wait(0.2)

            # 下一页
            if mouse.isPressedIn(self.next_button) and (self.current_page + 1) * self.files_per_page < len(
                    self.file_list):
                self.current_page += 1
                self.update_buttons_for_current_page()
                core.wait(0.2)

        return self.return_to_main