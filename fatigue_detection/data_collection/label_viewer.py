from psychopy import visual, event, core
from psychopy.visual import ButtonStim
import os
import re


class LabelViewerUI:
    def __init__(self, win):
        self.win = win
        self.return_to_main = False
        self.folder_path = "data_collection_label"
        self.file_list = []
        self.text_components = []
        self.buttons = []
        self.current_page = 0
        self.files_per_page = 10  # 每页最多显示 10 个文件

        # UI 组件初始化
        self.title = visual.TextStim(self.win, text="标签列表信息", pos=(0, 0.85), color="white", height=0.08, font='Microsoft YaHei')  # 标题
        self.back_button = ButtonStim(self.win, text="主页", pos=(-0.7, 0.7), size=(0.25, 0.08), font='Microsoft YaHei')  # 返回按钮
        self.no_file_text = visual.TextStim(self.win, text="无法找到数据文件。", pos=(0, 0), color="red", height=0.08, font='Microsoft YaHei')
        # 分页按钮
        self.prev_button = ButtonStim(self.win, text="上一页", pos=(-0.35, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')
        self.next_button = ButtonStim(self.win, text="下一页", pos=(-0.05, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')
        self.back_previous_button = ButtonStim(self.win, text="返回", pos=(0.25, -0.6), size=(0.15, 0.08), font='Microsoft YaHei')
        self.init_ui()

    def init_ui(self):
        # 加载文件
        if os.path.exists(self.folder_path) and os.path.isdir(self.folder_path):
            self.file_list = sorted([f for f in os.listdir(self.folder_path) if f.endswith(".txt")])
        else:
            self.file_list = []

        if self.file_list:
            self.update_buttons_for_current_page()

    def update_buttons_for_current_page(self):
        """根据当前页码更新按钮"""
        self.buttons.clear()
        y_start = 0.6
        spacing = -0.12

        start_idx = self.current_page * self.files_per_page
        end_idx = min(start_idx + self.files_per_page, len(self.file_list))

        for idx, filename in enumerate(self.file_list[start_idx:end_idx]):
            x_pos = -0.05  # 每个文件左侧位置
            y_pos = y_start + idx * spacing
            btn = ButtonStim(self.win, text=filename, pos=(x_pos, y_pos), size=(1.0, 0.08))
            self.buttons.append(btn)

    def show(self):
        mouse = event.Mouse(win=self.win)
        while True:
            self.title.draw()
            self.back_button.draw()

            if not self.file_list:
                self.no_file_text.draw()
            else:
                for btn in self.buttons:
                    btn.draw()
                self.prev_button.draw()
                self.next_button.draw()
                self.back_previous_button.draw()
            self.win.flip()

            keys = event.getKeys()
            if 'escape' in keys:
                self.return_to_main = True  # 返回主页
                break

            if mouse.isPressedIn(self.back_button):
                self.return_to_main = True  # 返回主页
                break

            if self.file_list:
                for idx, btn in enumerate(self.buttons):
                    if mouse.isPressedIn(btn):
                        self.show_file_content(self.file_list[self.current_page * self.files_per_page + idx])
                        core.wait(0.2)
                        break

            # 点击上一页需要当前页大于0
            if mouse.isPressedIn(self.prev_button) and self.current_page > 0:
                self.current_page -= 1
                self.update_buttons_for_current_page()
                core.wait(0.2)
            # 点击下一页需要并且下一页仍有内容可以显示时，才允许翻页
            elif mouse.isPressedIn(self.next_button) and (self.current_page + 1) * self.files_per_page < len(
                    self.file_list):
                self.current_page += 1
                self.update_buttons_for_current_page()
                core.wait(0.2)
            elif mouse.isPressedIn(self.back_previous_button):
                self.return_to_main = False  # 返回上级界面，不返回主页
                break

        return self.return_to_main

    def show_file_content(self, filename):
        """打开指定文件并显示结构化内容"""
        file_path = os.path.join(self.folder_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            lines = [f"无法读取文件：{str(e)}"]

        # 解析日志内容
        table_data = self._parse_log_content(lines)

        if not table_data:
            self._show_empty_message()
            return

        # 展示带分页的表格
        self._show_table_with_pagination(table_data, filename)

    def _parse_log_content(self, lines):
        """解析日志内容，返回结构化数据"""

        table_data = []
        for line in lines:
            line = line.strip()
            record_type = "Unknown"
            fatigue_level = "—"
            time = None  # "Record Type", "Fatigue Level", "Time", "Note"
            note = ""
            if not line:
                continue

            if line.startswith("Data collection started at:"):
                record_type = "采集开始"
                time = line.split(": ", 1)[1]
            elif line.startswith("Data collection ended by user at:"):
                record_type = "采集结束"
                time = line.split(": ", 1)[1]
            elif line.startswith("Prompt shown at:"):
                record_type = "弹窗提示"
                time = line.split(": ", 1)[1]
            elif line.startswith("User input fatigue level:"):
                record_type = "用户输入"
                parts = line.split(" ")
                fatigue_level = int(parts[4])
                time = " ".join(parts[6:8]).rstrip('.')
                note = parts[8].strip('()') if len(parts) > 8 else ""
            if fatigue_level == 1:
                fatigue_level = "1-清醒"
            elif fatigue_level == 2:
                fatigue_level = "2-疲劳"
            elif fatigue_level == 3:
                fatigue_level = "3-昏睡"
            # elif fatigue_level == 4:
            #     fatigue_level = "4-重度疲劳"
            table_data.append([record_type, fatigue_level, time.split('.')[0], note])  # 去掉毫秒
        return table_data

    def _show_empty_message(self):
        """显示空文件提示"""
        from psychopy import visual, event

        content_text = visual.TextStim(self.win, text="文件为空", pos=(0, 0), color="white", height=0.05, font='Microsoft YaHei')
        back_button = ButtonStim(self.win, text="返回", pos=(0, -0.45), size=(0.2, 0.08), font='Microsoft YaHei')
        mouse = event.Mouse(win=self.win)

        while True:
            content_text.draw()
            back_button.draw()
            self.win.flip()
            if mouse.isPressedIn(back_button):
                return

    def _create_stimuli_for_page(self, table_data, page):
        """根据当前页码生成对应的 TextStim 对象"""
        letter_height = 0.04
        y_start = 0.4
        line_spacing = -letter_height * 2

        col_widths = [0.3, 0.3, 0.3, 0.3]  # 合计 = 1.2
        total_width = sum(col_widths)

        col_positions = []
        current_x = -total_width / 2 + col_widths[0] / 2
        for width in col_widths:
            col_positions.append(current_x)
            current_x += width

        items_per_page = 10
        start_idx = page * items_per_page
        end_idx = min(start_idx + items_per_page, len(table_data))

        header_stims = []
        row_stims = []

        # 表头
        headers = ["时间类型", "状态类型", "时间", "备注"]
        for i, header in enumerate(headers):
            stim = visual.TextStim(
                self.win,
                text=header,
                pos=(col_positions[i], y_start),
                color='yellow',
                height=letter_height,
                alignText='center',
                font='Microsoft YaHei'
            )
            header_stims.append(stim)

        # 数据行
        for row_idx, row in enumerate(table_data[start_idx:end_idx]):
            row_line = []
            for col_idx, text in enumerate(row):
                x_pos = col_positions[col_idx]
                y_pos = y_start + (row_idx + 1) * line_spacing
                stim = visual.TextStim(
                    self.win,
                    text=text,
                    pos=(x_pos, y_pos),
                    color='white',
                    height=letter_height,
                    alignText='center'
                )
                row_line.append(stim)
            row_stims.append(row_line)

        return header_stims, row_stims

    def _show_table_with_pagination(self, table_data, filename=None):
        """展示结构化表格并支持翻页"""

        items_per_page = 10
        current_page = 0

        file_info_text = visual.TextStim(self.win, text=f"当前文件：{filename}", pos=(0, 0.7),
                                         color='lime', height=0.045, alignText='center', font='Microsoft YaHei')
        # 按钮
        back_previous_button = ButtonStim(self.win, text="返回", pos=(0, -0.7), size=(0.2, 0.08), font='Microsoft YaHei')
        prev_button = ButtonStim(self.win, text="上一页", pos=(-0.2, -0.7), size=(0.15, 0.08), font='Microsoft YaHei')
        next_button = ButtonStim(self.win, text="下一页", pos=(0.2, -0.7), size=(0.15, 0.08), font='Microsoft YaHei')

        mouse = event.Mouse(win=self.win)

        while True:
            header_stims, row_stims = self._create_stimuli_for_page(table_data, current_page)

            # 文件名提示文本
            if filename:
                file_info_text.draw()
            # 绘制组件
            for h in header_stims:
                h.draw()
            for row in row_stims:
                for cell in row:
                    cell.draw()
            back_previous_button.draw()
            prev_button.draw()
            next_button.draw()
            self.win.flip()

            # 处理事件
            if mouse.isPressedIn(back_previous_button):
                return

            if mouse.isPressedIn(prev_button) and current_page > 0:
                current_page -= 1
                core.wait(0.2)

            if mouse.isPressedIn(next_button) and (current_page + 1) * items_per_page < len(table_data):
                current_page += 1
                core.wait(0.2)
