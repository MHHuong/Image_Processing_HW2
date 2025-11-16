import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import image_processing as ip
import numpy as np
import cv2

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing")
        self.root.geometry("800x700")
        self.original_image = None
        self.processed_image = None
        self.tk_image_top = None
        self.tk_image_bottom = None
        self.sliders = {}
        self.create_widgets()

    def create_widgets(self):
        left_panel = tk.Frame(self.root, width=400, height=600, bg="lightgray")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas_top = tk.Canvas(left_panel, width=350, height=250, bg="black")
        self.canvas_top.pack(pady=10)
        self.canvas_bottom = tk.Canvas(left_panel, width=350, height=250, bg="black")
        self.canvas_bottom.pack(pady=10)
        right_panel = tk.Frame(self.root, width=350, height=600, bg="white", relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_negative = tk.Button(right_panel, text="Negative image", command=self.run_negative)
        btn_negative.pack(pady=5, padx=10, anchor=tk.W)

        lbl_tool_title = tk.Label(right_panel, text="CÔNG CỤ BIẾN ĐỔI", font=("Arial", 12, "bold"))
        lbl_tool_title.pack(pady=5)

        log_sliders, log_frame = self.create_transformation_section(
            right_panel,
            "Biến đổi Log",
            ['C', 'Logarit'],
            ["#1E90FF", "#87CEFA", "#ADD8E6"],
            command=lambda val: self.on_log_params_change()
        )
        self.sliders['log_c'] = log_sliders[0]
        self.sliders['log_logarit'] = log_sliders[1]

        piecewise_sliders, piece_frame = self.create_transformation_section(
            right_panel, "Biến đổi Piecewise-Linear",
            ['Cao', 'Thấp'],
            ["#800080", "#DA70D6", "#EE82EE"],
            command=lambda val: self.on_piecewise_params_change()
        )
        self.sliders['piece_high'] = piecewise_sliders[0]
        self.sliders['piece_low'] = piecewise_sliders[1]

        gamma_sliders, gamma_frame = self.create_transformation_section(
            right_panel,
            "Biến đổi Gamma",
            ['C', 'Gamma'],
            ["#FF4500", "#FFA07A", "#FF7F50"],
            command=lambda val: self.run_gamma_transform())
        self.sliders['gamma_c'] = gamma_sliders[0]
        self.sliders['gamma_gamma'] = gamma_sliders[1]

        matrix_sliders, matrix_frame = self.create_transformation_section(
            right_panel,
            "Avg filter",
            ['n'],
            ["#1db353", "#03fc5e", "#79f2a5"],
            command=lambda val: self.run_avg_filter())
        self.sliders['n'] = matrix_sliders[0]

        gauss_sliders, gauss_frame = self.create_transformation_section(
            right_panel,
            "Gauss filter",
            ['l', 'sigma'],
            ["#1db353", "#03fc5e", "#79f2a5"],
            command=lambda val: self.run_avg_filter())
        self.sliders['l'] = gauss_sliders[0]
        self.sliders['sigma'] = gauss_sliders[1]

        button_frame = tk.Frame(right_panel, pady=10)
        button_frame.pack(pady=10)
        btn_choose_image = tk.Button(button_frame, text="Chọn ảnh", command=self.open_image)
        btn_choose_image.pack(side=tk.LEFT, padx=5)
        btn_save = tk.Button(button_frame, text="Lưu ra file", command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=5)
        btn_close = tk.Button(button_frame, text="Close", command=self.root.destroy)
        btn_close.pack(side=tk.LEFT, padx=5)

    def create_transformation_section(self, parent, title, param_names, colors, command=None):
        frame = tk.Frame(parent, bg=colors[0], padx=5, pady=5, relief=tk.GROOVE, borderwidth=1)
        frame.pack(fill=tk.X, pady=5, padx=10)

        lbl_title = tk.Label(frame, text=title, bg=colors[0], font=("Arial", 10, "bold"))
        lbl_title.pack(anchor=tk.W)

        sliders = []
        for name in param_names:
            slider_frame = tk.Frame(frame, bg=colors[0])
            slider_frame.pack(fill=tk.X)

            lbl_param = tk.Label(slider_frame, text=f"Hệ số {name}", bg=colors[0])
            lbl_param.pack(side=tk.LEFT)

            from_val, to_val, res = 0, 100, 1

            if name.lower() in ['c', 'gamma']:
                from_val, to_val, res = 0, 20, 0.1

            if name.lower() == 'logarit':
                from_val, to_val, res = 2, 20, 1

            if name.lower() in ['cao', 'thấp']:
                from_val, to_val, res = 0, 255, 1

            if name.lower() == 'n':
                from_val, to_val, res = 1, 51, 1

            if name.lower() in ['l', 'sigma']:
                from_val, to_val, res = 1, 20, 0.1

            slider = tk.Scale(
                slider_frame,
                from_=from_val, to_=to_val,
                orient=tk.HORIZONTAL, length=200,
                bg=colors[1], troughcolor=colors[2],
                resolution=res,
                command=command
            )
            if name.lower() in ['c', 'gamma']:
                slider.set(1)
            if name.lower() == 'logarit':
                slider.set(2)

            slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            sliders.append(slider)

        return sliders, frame

    def create_smoothing_section(self, parent, title, num_sliders, colors):
        frame = tk.Frame(parent, bg=colors[0], padx=5, pady=5, relief=tk.GROOVE, borderwidth=1)
        frame.pack(fill=tk.X, pady=5, padx=10)
        lbl_title = tk.Label(frame, text=title, bg=colors[0], font=("Arial", 10, "bold"))
        lbl_title.pack(anchor=tk.W)
        sliders = []
        for i in range(num_sliders):
            slider_frame = tk.Frame(frame, bg=colors[0])
            slider_frame.pack(fill=tk.X)
            if i == 0:
                lbl_param = tk.Label(slider_frame, text="Kích thước lọc", bg=colors[0])
            else:
                lbl_param = tk.Label(slider_frame, text="Hệ số Sigma", bg=colors[0])
            lbl_param.pack(side=tk.LEFT)
            slider = tk.Scale(slider_frame, from_=0, to_=21, orient=tk.HORIZONTAL, length=200, resolution=0.1,
                              bg=colors[1], troughcolor=colors[2])
            slider.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            sliders.append(slider)
        return sliders

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            self.original_image = Image.open(file_path)
            self.processed_image = self.original_image.copy()
            img_top = self.original_image.copy()
            img_top.thumbnail((350, 250))
            self.tk_image_top = ImageTk.PhotoImage(img_top)
            self.canvas_top.delete("all")
            self.canvas_top.create_image(175, 125, anchor=tk.CENTER, image=self.tk_image_top)
            self.update_canvas_bottom(self.processed_image)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở file ảnh: {e}")

    def update_canvas_bottom(self, pil_image):
        img_bottom = pil_image.copy()
        img_bottom.thumbnail((350, 250))

        self.tk_image_bottom = ImageTk.PhotoImage(img_bottom)
        self.canvas_bottom.delete("all")
        self.canvas_bottom.create_image(175, 125, anchor=tk.CENTER, image=self.tk_image_bottom)

    def run_negative(self):
        if not self.original_image:
            return

        self.processed_image = ip.apply_negative(self.original_image)
        self.update_canvas_bottom(self.processed_image)

    def run_log_transform(self):
        if not self.original_image:
            return
        try:
            c_val = self.sliders['log_c'].get()
            logarit_val = self.sliders['log_logarit'].get()
            self.processed_image = ip.apply_log(self.original_image, c_val, logarit_val)
            self.update_canvas_bottom(self.processed_image)

        except Exception as e:
            messagebox.showerror(e)

    def run_gamma_transform(self):
        if not self.original_image:
            return
        try:
            c_val = self.sliders['gamma_c'].get()
            gamma_val = self.sliders['gamma_gamma'].get()
            self.processed_image = ip.apply_gamma(self.original_image, c_val, gamma_val)
            self.update_canvas_bottom(self.processed_image)

        except Exception as e:
            messagebox.showerror(e)

    def run_avg_filter(self):
        if not self.original_image:
            return
        try:
            n_val = int(self.sliders['n'].get())
            if n_val < 1:
                n_val = 1
            img_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            filtered = ip.apply_avg_filter(img_cv, n_val)
            filtered_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
            self.processed_image = Image.fromarray(filtered_rgb)
            self.update_canvas_bottom(self.processed_image)
        except Exception as e:
            print(e)
    def save_image(self):
        if not self.processed_image:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.processed_image.save(file_path)
                messagebox.showinfo("Thành công", f"Đã lưu ảnh tại: {file_path}")
            except Exception as e:
                messagebox.showerror(e)

    def on_log_params_change(self, *_):
        if not self.original_image:
            return
        self._recalc_log_c_max()
        self.run_log_transform()

    def _recalc_log_c_max(self):
        if not self.original_image:
            return
        base = float(self.sliders['log_logarit'].get())
        if base <= 1:
            base = np.e
        arr = np.array(self.original_image.convert('RGB'), dtype=np.float64)
        rmax = float(arr.max())
        if rmax < 1:
            rmax = 1.0
        c_max = 255.0 / (np.log(1.0 + rmax) / np.log(base))
        c_slider = self.sliders['log_c']
        c_slider.configure(to=c_max)
        if c_slider.get() > c_max:
            c_slider.set(c_max)

    def on_gamma_params_change(self, *_):
        if not self.original_image:
            return
        self.run_gamma_transform()

    def on_piecewise_params_change(self, *_):
        if not self.original_image:
            return
        r_high = int(self.sliders['piece_high'].get())
        r_low = int(self.sliders['piece_low'].get())
        if r_low >= r_high:
            if r_low > 0:
                r_low = r_high - 1
            else:
                r_high = r_low + 1
            self.sliders['piece_low'].set(r_low)
            self.sliders['piece_high'].set(r_high)

        self.processed_image = ip.apply_piecewise_linear(self.original_image, r_low, r_high)
        self.update_canvas_bottom(self.processed_image)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
