import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter, ImageEnhance
import customtkinter as ctk
import image_processing as ip
import numpy as np
import cv2

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing")
        self.root.geometry("1200x800")
        self.original_image = None
        self.processed_image = None
        self.tk_image_top = None
        self.tk_image_bottom = None
        self.ctkimg_top = None
        self.ctkimg_bottom = None
        self.sliders = {}
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.create_widgets()

    def create_widgets(self):
        # Set background color instead of overlay image
        self.root.configure(bg="#0d0d0d")

        # Main layout using grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=4)
        self.root.grid_columnconfigure(1, weight=1)

        left_panel = ctk.CTkFrame(self.root, fg_color=("#1a1a1a", "#1a1a1a"))
        left_panel.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        left_panel.grid_rowconfigure(0, weight=0)
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        title_bar = ctk.CTkLabel(left_panel, text="Image Processing", font=("Segoe UI Semibold", 20))
        title_bar.grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))

        images_container = ctk.CTkFrame(left_panel, fg_color=("#101010", "#101010"))
        images_container.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        images_container.grid_columnconfigure(0, weight=1)
        images_container.grid_rowconfigure((0, 1), weight=1)

        self.img_top_label = ctk.CTkLabel(images_container, text="Ảnh gốc", corner_radius=8, font=("Segoe UI", 12))
        self.img_top_label.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))
        self.img_bottom_label = ctk.CTkLabel(images_container, text="Ảnh xử lý", corner_radius=8, font=("Segoe UI", 12))
        self.img_bottom_label.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))

        right_panel = ctk.CTkFrame(self.root, fg_color=("#0f0f0f", "#0f0f0f"))
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=12)
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(0, weight=0)
        right_panel.grid_rowconfigure(1, weight=0)
        right_panel.grid_rowconfigure(2, weight=1)
        right_panel.grid_rowconfigure(3, weight=0)

        top_buttons = ctk.CTkFrame(right_panel, fg_color="transparent")
        top_buttons.grid(row=0, column=0, sticky="ew", padx=12, pady=(8, 4))
        btn_choose_image = ctk.CTkButton(top_buttons, text="Chọn ảnh", command=self.open_image)
        btn_choose_image.pack(side=tk.LEFT, padx=(0, 8))
        btn_negative = ctk.CTkButton(top_buttons, text="Negative image", command=self.run_negative)
        btn_negative.pack(side=tk.LEFT)

        # Tabview for organizing tools
        self.tabview = ctk.CTkTabview(right_panel, fg_color=("#0f0f0f", "#0f0f0f"))
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=12, pady=(8, 8))
        
        tab_transform = self.tabview.add("Biến đổi")
        tab_filter = self.tabview.add("Filter")
        
        scroll_transform = ctk.CTkScrollableFrame(tab_transform, fg_color="transparent", height=400)
        scroll_transform.pack(fill="x", padx=0, pady=0)
        
        scroll_filter = ctk.CTkScrollableFrame(tab_filter, fg_color="transparent", height=400)
        scroll_filter.pack(fill="x", padx=0, pady=0)

        log_sliders, log_frame = self.create_transformation_section(
            scroll_transform,
            "Biến đổi Log",
            ['C', 'Logarit'],
            command=lambda val: self.on_log_params_change()
        )
        self.sliders['log_c'] = log_sliders[0]
        self.sliders['log_logarit'] = log_sliders[1]

        piecewise_sliders, piece_frame = self.create_transformation_section(
            scroll_transform, "Biến đổi Piecewise-Linear",
            ['Cao', 'Thấp'],
            command=lambda val: self.on_piecewise_params_change()
        )
        self.sliders['piece_high'] = piecewise_sliders[0]
        self.sliders['piece_low'] = piecewise_sliders[1]

        gamma_sliders, gamma_frame = self.create_transformation_section(
            scroll_transform,
            "Biến đổi Gamma",
            ['C', 'Gamma'],
            command=lambda val: self.run_gamma_transform())
        self.sliders['gamma_c'] = gamma_sliders[0]
        self.sliders['gamma_gamma'] = gamma_sliders[1]

        matrix_sliders, matrix_frame = self.create_transformation_section(
            scroll_filter,
            "Avg filter",
            ['n'],
            command=lambda val: self.run_avg_filter())
        self.sliders['n'] = matrix_sliders[0]

        gauss_sliders, gauss_frame = self.create_transformation_section(
            scroll_filter,
            "Gauss filter",
            ['l', 'sigma'],
            command=lambda val: self.run_avg_filter())
        self.sliders['l'] = gauss_sliders[0]
        self.sliders['sigma'] = gauss_sliders[1]

        bottom_bar = ctk.CTkFrame(right_panel, fg_color="transparent")
        bottom_bar.grid(row=3, column=0, sticky="ew", padx=12, pady=(8, 12))
        btn_apply = ctk.CTkButton(bottom_bar, text="Áp dụng", command=self.apply_changes, fg_color="#28a745", hover_color="#218838")
        btn_apply.pack(side=tk.LEFT, padx=(0, 8))
        btn_save = ctk.CTkButton(bottom_bar, text="Lưu ra file", command=self.save_image)
        btn_save.pack(side=tk.LEFT, padx=(0, 8))
        btn_close = ctk.CTkButton(bottom_bar, text="Đóng", command=self.root.destroy)
        btn_close.pack(side=tk.LEFT)

    def create_transformation_section(self, parent, title, param_names, command=None):
        frame = ctk.CTkFrame(parent, fg_color=("#151515", "#151515"))
        frame.pack(fill="x", padx=12, pady=8)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=0)

        lbl_title = ctk.CTkLabel(frame, text=title, font=("Segoe UI", 12, "bold"))
        lbl_title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(8, 6))

        sliders = []
        row = 1
        for name in param_names:
            lbl_param = ctk.CTkLabel(frame, text=f"Hệ số {name}")
            lbl_param.grid(row=row, column=0, sticky="w", padx=8)

            from_val, to_val, res = 0, 100, 1
            n = name.lower()
            if n in ['c', 'gamma']:
                from_val, to_val, res = 0, 20, 0.1
            elif n == 'logarit':
                from_val, to_val, res = 2, 20, 1
            elif n in ['cao', 'thap', 'thấp']:
                from_val, to_val, res = 0, 255, 1
            elif n == 'n':
                from_val, to_val, res = 1, 51, 1
            elif n in ['l', 'sigma']:
                from_val, to_val, res = 1, 20, 0.1

            value_label = ctk.CTkLabel(frame, text="0.0", width=50, font=("Segoe UI", 11))
            value_label.grid(row=row, column=2, sticky="e", padx=(4, 8))

            def make_slider_command(val_lbl, orig_cmd, resolution):
                def wrapper(value):
                    if resolution >= 1:
                        val_lbl.configure(text=f"{int(float(value))}")
                    else:
                        val_lbl.configure(text=f"{float(value):.2f}")
                    if orig_cmd:
                        orig_cmd(value)
                return wrapper

            slider = ctk.CTkSlider(
                frame,
                from_=from_val,
                to=to_val,
                number_of_steps=int((to_val - from_val) / res) if res != 0 else 0,
                command=make_slider_command(value_label, command, res),
                fg_color="#FFFFFF",          
                progress_color="#FFFFFF",    
                button_color="#FFFFFF",       
                button_hover_color="#EDEDED",
            )

            if n in ['c', 'gamma']:
                slider.set(1)
                value_label.configure(text="1.00")
            elif n == 'logarit':
                slider.set(2)
                value_label.configure(text="2")
            else:
                if res >= 1:
                    value_label.configure(text=f"{int(slider.get())}")
                else:
                    value_label.configure(text=f"{slider.get():.2f}")

            slider.grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=6)
            sliders.append(slider)
            row += 1

        return sliders, frame

    def create_smoothing_section(self, parent, title, num_sliders, colors):
        frame = ctk.CTkFrame(parent)
        frame.pack(fill="x", padx=12, pady=8)
        sliders = []
        return sliders

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        try:
            self.original_image = Image.open(file_path)
            self.processed_image = self.original_image.copy()
            img_top = self.original_image.copy()
            self.update_top_image(img_top)
            self.update_canvas_bottom(self.processed_image)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở file ảnh: {e}")

    def update_canvas_bottom(self, pil_image):
        area = self._label_inner_size(self.img_bottom_label)
        if area[0] <= 0 or area[1] <= 0:
            w, h = 600, 350
        else:
            w, h = area
        img = pil_image.copy()
        img.thumbnail((w, h))
        self.ctkimg_bottom = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.img_bottom_label.configure(image=self.ctkimg_bottom, text="")

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
    def apply_changes(self):
        """Apply processed image as new original image for further editing"""
        if not self.processed_image:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh xử lý để áp dụng")
            return
        
        # Save processed image as new original
        self.original_image = self.processed_image.copy()
        
        # Update top image display
        self.update_top_image(self.original_image)
        
        # Reset sliders to default values
        if 'log_c' in self.sliders:
            self.sliders['log_c'].set(1)
        if 'log_logarit' in self.sliders:
            self.sliders['log_logarit'].set(2)
        if 'gamma_c' in self.sliders:
            self.sliders['gamma_c'].set(1)
        if 'gamma_gamma' in self.sliders:
            self.sliders['gamma_gamma'].set(1)
        if 'piece_low' in self.sliders:
            self.sliders['piece_low'].set(50)
        if 'piece_high' in self.sliders:
            self.sliders['piece_high'].set(200)
        if 'n' in self.sliders:
            self.sliders['n'].set(3)
        if 'l' in self.sliders:
            self.sliders['l'].set(3)
        if 'sigma' in self.sliders:
            self.sliders['sigma'].set(1)
        
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
        try:
            c_slider.configure(to=c_max, number_of_steps=max(1, int((c_max - 0) / 0.1)))
        except Exception:
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

    def _label_inner_size(self, label_widget):
        try:
            w = max(0, label_widget.winfo_width() - 24)
            h = max(0, label_widget.winfo_height() - 24)
            return w, h
        except Exception:
            return 600, 350

    def update_top_image(self, pil_image):
        area = self._label_inner_size(self.img_top_label)
        if area[0] <= 0 or area[1] <= 0:
            w, h = 600, 350
        else:
            w, h = area
        img = pil_image.copy()
        img.thumbnail((w, h))
        self.ctkimg_top = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.img_top_label.configure(image=self.ctkimg_top, text="")


if __name__ == "__main__":
    root = ctk.CTk()
    app = ImageApp(root)
    root.mainloop()
