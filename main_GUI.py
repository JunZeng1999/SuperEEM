import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import numpy as np
import matplotlib.pyplot as plt
from niceplots import parula
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

try:
    from model.model_SuperEEM import FluorescenceSRUNet

    MODELS_IMPORTED = True
except ImportError:
    MODELS_IMPORTED = False
    print("Warning: model_SuperEEM.py not found. Ensure it is in the same directory.")


class FluorescenceSRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SuperEEM")

        self.root.geometry("1350x830")
        self.root.minsize(1150, 700)

        self.root.configure(bg='#f5f6f8')
        self.style = ttk.Style()
        self.style.theme_use('clam')

        font_main = ('Segoe UI', 10)
        font_bold = ('Segoe UI', 10, 'bold')
        font_title = ('Segoe UI', 11, 'bold')

        self.style.configure('Main.TFrame', background='#f5f6f8')
        self.style.configure('Panel.TFrame', background='#ffffff')
        self.style.configure('Academic.TLabelframe', background='#ffffff', bordercolor='#dcdfe6', padding=10)
        self.style.configure('Academic.TLabelframe.Label', background='#ffffff', foreground='#1f2d3d', font=font_title)

        self.style.configure('TLabel', background='#ffffff', foreground='#475669', font=font_main)
        self.style.configure('Step.TLabel', background='#ffffff', foreground='#20a0ff', font=font_bold)
        self.style.configure('StatusText.TLabel', background='#ffffff', font=('Segoe UI', 9, 'italic'))

        self.style.configure('Action.TButton', font=font_main, background='#f0f2f5', foreground='#475669',
                             borderwidth=1, bordercolor='#dcdfe6')
        self.style.map('Action.TButton', background=[('active', '#e4e7ed'), ('disabled', '#f5f7fa')],
                       foreground=[('disabled', '#c0c4cc')])
        self.style.configure('Run.TButton', font=font_bold, background='#20a0ff', foreground='#ffffff', borderwidth=0)
        self.style.map('Run.TButton', background=[('active', '#1d8ce0'), ('disabled', '#a0cfff')],
                       foreground=[('disabled', '#ffffff')])
        self.style.configure('Export.TButton', font=font_bold, background='#13ce66', foreground='#ffffff',
                             borderwidth=0)
        self.style.map('Export.TButton', background=[('active', '#11b95c'), ('disabled', '#85e7a9')],
                       foreground=[('disabled', '#ffffff')])
        self.style.configure('TCombobox', font=font_main)

        self.style.configure('Status.TLabel', background='#ffffff', foreground='#5e6d82', font=('Consolas', 9),
                             anchor=tk.W, padding=(15, 6))

        plt.rcParams.update({
            'axes.facecolor': '#ffffff',
            'figure.facecolor': '#ffffff',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.0,
            'axes.labelcolor': '#222222',
            'axes.titlecolor': '#111111',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'grid.color': '#f5f5f5',
            'text.color': '#000000',
            'font.size': 9,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Segoe UI']
        })

        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.file_paths = []
        self.data_registry = {}
        self.active_key = None

        self.ex_slider_val = tk.IntVar(value=0)
        self.em_slider_val = tk.IntVar(value=0)

        self._setup_ui()

    def _setup_ui(self):
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(15, 10))

        header_frame = ttk.Frame(main_container, style='Main.TFrame')
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        ttk.Label(header_frame, text="SuperEEM Workspace", font=('Segoe UI', 16, 'bold'), foreground='#1f2d3d',
                  background='#f5f6f8').pack(side=tk.LEFT)
        ttk.Label(header_frame, text="— Deep Learning-Based EEM Super-Resolution Reconstruction Toolkit",
                  font=('Segoe UI', 11), foreground='#5e6d82', background='#f5f6f8').pack(side=tk.LEFT, padx=10,
                                                                                          pady=(4, 0))

        body_frame = ttk.Frame(main_container, style='Main.TFrame')
        body_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(body_frame, width=350, style='Panel.TFrame')
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)

        model_box = ttk.LabelFrame(left_panel, text="Model Configuration", style='Academic.TLabelframe')
        model_box.pack(fill=tk.X, anchor=tk.N, pady=(5, 5))
        ttk.Label(model_box, text="Step 1: Load Model Weights", style='Step.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.btn_load_model = ttk.Button(model_box, text="Browse .pth File", command=self.load_model_file,
                                         style='Action.TButton')
        self.btn_load_model.pack(fill=tk.X, pady=5)
        self.lbl_model_status = ttk.Label(model_box, text="Status: Model uninitialized", style='StatusText.TLabel',
                                          foreground='#f56c6c')
        self.lbl_model_status.pack(anchor=tk.W, pady=(2, 0))

        data_box = ttk.LabelFrame(left_panel, text="Data Input", style='Academic.TLabelframe')
        data_box.pack(fill=tk.BOTH, expand=True, anchor=tk.N, pady=4)
        ttk.Label(data_box, text="Step 2: Import Low-Resolution EEM", style='Step.TLabel').pack(anchor=tk.W, pady=(0, 2))

        self.btn_load_lr = ttk.Button(data_box, text="📂 Select .txt EEM (Support Multi-Select)",
                                      command=self.load_lr_txt_batch, style='Action.TButton', state=tk.DISABLED)
        self.btn_load_lr.pack(fill=tk.X, pady=2)

        ttk.Separator(data_box, orient='horizontal').pack(fill=tk.X, pady=6)
        ttk.Label(data_box, text="Target File Selector:", font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W,
                                                                                                    pady=(2, 2))

        self.cb_file_selector = ttk.Combobox(data_box, state="disabled", textvariable=tk.StringVar())
        self.cb_file_selector.pack(fill=tk.X, pady=2)
        self.cb_file_selector.bind("<<ComboboxSelected>>", self.on_dropdown_switch)

        slice_box = ttk.LabelFrame(left_panel, text="Wavelength Profile Interactive Locator",
                                   style='Academic.TLabelframe')
        slice_box.pack(fill=tk.X, anchor=tk.S, pady=4)

        self.lbl_slider_ex = ttk.Label(slice_box, text="Excitation (EX) Wavelength: -- nm",
                                       font=('Segoe UI', 9, 'bold'))
        self.lbl_slider_ex.pack(anchor=tk.W)
        self.slider_ex = tk.Scale(slice_box, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.ex_slider_val,
                                  command=lambda e: self.replotted_canvas(), bg='#ffffff', bd=0, highlightthickness=0,
                                  activebackground='#20a0ff')
        self.slider_ex.pack(fill=tk.X, pady=(0, 5))

        self.lbl_slider_em = ttk.Label(slice_box, text="Emission (EM) Wavelength: -- nm", font=('Segoe UI', 9, 'bold'))
        self.lbl_slider_em.pack(anchor=tk.W)
        self.slider_em = tk.Scale(slice_box, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.em_slider_val,
                                  command=lambda e: self.replotted_canvas(), bg='#ffffff', bd=0, highlightthickness=0,
                                  activebackground='#20a0ff')
        self.slider_em.pack(fill=tk.X, pady=(0, 2))

        proc_box = ttk.LabelFrame(left_panel, text="Execution", style='Academic.TLabelframe')
        proc_box.pack(fill=tk.X, anchor=tk.S, pady=(4, 2))

        self.btn_sr = ttk.Button(proc_box, text="▶ Run Reconstruction", command=self.run_all_super_resolution,
                                 style='Run.TButton', state=tk.DISABLED)
        self.btn_sr.pack(fill=tk.X, ipady=3, pady=2)

        self.btn_export = ttk.Button(proc_box, text="💾 Export Current SR EEM", command=self.export_to_txt,
                                     style='Export.TButton', state=tk.DISABLED)
        self.btn_export.pack(fill=tk.X, ipady=2, pady=2)

        right_panel = ttk.Frame(body_frame, style='Panel.TFrame')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        display_border = tk.Frame(right_panel, bg='#eaedf2', bd=1)
        display_border.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig, self.axes = plt.subplots(2, 2, figsize=(9, 7.2))
        self.ax_lr = self.axes[0, 0]
        self.ax_sr = self.axes[0, 1]
        self.ax_ex_slice = self.axes[1, 0]
        self.ax_em_slice = self.axes[1, 1]

        self.fig.subplots_adjust(wspace=0.25, hspace=0.35, top=0.94, bottom=0.08, left=0.08, right=0.96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_border)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._clear_axes()

        status_frame = tk.Frame(self.root, bg='#ffffff', bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_bar = ttk.Label(status_frame,
                                    text="System Ready. Please load pre-trained network weights to begin.",
                                    style='Status.TLabel')
        self.status_bar.pack(fill=tk.X)

    def _clear_axes(self):
        for ax in self.axes.flatten():
            ax.clear()
        self.ax_lr.set_box_aspect(1)
        self.ax_sr.set_box_aspect(1)
        self.ax_lr.set_title("Low-Resolution Input (LR-EEM)", fontsize=10, fontweight='bold', pad=6)
        self.ax_lr.set_xlabel("Excitation Wavelength (nm)", labelpad=4)
        self.ax_lr.set_ylabel("Emission Wavelength (nm)", labelpad=4)
        self.ax_sr.set_title("Super-Resolved Output (SR-EEM)", fontsize=10, fontweight='bold', pad=6)
        self.ax_sr.set_xlabel("Excitation Wavelength (nm)", labelpad=4)
        self.ax_sr.set_ylabel("Emission Wavelength (nm)", labelpad=4)
        self.ax_ex_slice.set_title("Emission Profile Cut-line", fontsize=9, pad=6)
        self.ax_em_slice.set_title("Excitation Profile Cut-line", fontsize=9, pad=6)
        self.canvas.draw()

    def update_status(self, text):
        self.status_bar.config(text=f"Status: {text}")
        self.root.update_idletasks()

    def load_model_file(self):
        if not MODELS_IMPORTED:
            messagebox.showerror("Model Error", "The file 'model_SuperEEM.py' was not found in the working directory.")
            return

        file_path = filedialog.askopenfilename(title="Select Pre-trained Model Weights",
                                               filetypes=[("PyTorch Model Weights", "*.pth")])
        if not file_path:
            return

        self.update_status(f"Loading weights from {os.path.basename(file_path)}...")
        try:
            self.model = FluorescenceSRUNet(base_ch=64)
            state_dict = torch.load(file_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.lbl_model_status.config(text=f"Status: Loaded ({os.path.basename(file_path)})", foreground='#67c23a')
            self.btn_load_lr.config(state=tk.NORMAL)
            self.update_status("Weights loaded successfully. Ready to import EEM Text data.")
        except Exception as e:
            messagebox.showerror("Loading Failed", f"An error occurred while restoring model parameters:\n{str(e)}")
            self.lbl_model_status.config(text="Status: Loading Error", foreground='#f56c6c')

    def load_lr_txt_batch(self):
        files = filedialog.askopenfilenames(title="Select Low-Resolution EEM Data Text Files",
                                            filetypes=[("EEM Text Matrices", "*.txt")])
        if not files:
            return

        self.update_status("Parsing EEM text data matrices and structural wavelengths...")
        self.file_paths = list(files)
        self.data_registry.clear()
        dropdown_list = []

        for f_path in self.file_paths:
            f_name = os.path.basename(f_path)
            try:
                with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = [line.strip().split() for line in f if line.strip()]

                if not lines:
                    raise ValueError("Empty file.")

                first_row = [float(x) for x in lines[0]]
                data_rows = [[float(x) for x in row] for row in lines[1:]]

                if len(first_row) == len(data_rows[0]) - 1:
                    ex_wavelengths = np.array(first_row, dtype=np.float32)
                elif len(first_row) == len(data_rows[0]):
                    ex_wavelengths = np.array(first_row[1:], dtype=np.float32)
                else:
                    raise ValueError(
                        f"Row dimension mismatch. Row 1 has {len(first_row)} elements, but data rows have {len(data_rows[0])} elements.")

                em_wavelengths = np.array([row[0] for row in data_rows], dtype=np.float32)
                eem_matrix = np.array([row[1:] for row in data_rows], dtype=np.float32)

                import torch.nn.functional as F
                lr_tensor_vis = torch.from_numpy(eem_matrix).unsqueeze(0).unsqueeze(0)
                lr_128_tensor = F.interpolate(lr_tensor_vis, size=(128, 128), mode='nearest')
                lr_plot_data = lr_128_tensor.squeeze().numpy()

                eem_matrix_down = eem_matrix[::2, ::2]
                lr_tensor_vis1 = torch.from_numpy(eem_matrix_down).unsqueeze(0).unsqueeze(0)
                lr_128_tensor1 = F.interpolate(lr_tensor_vis1, size=(128, 128), mode='nearest')
                lr_plot_data1 = lr_128_tensor1.squeeze().numpy()

                sr_ex_waves = np.linspace(ex_wavelengths[0], ex_wavelengths[-1], 128, dtype=np.float32)
                sr_em_waves = np.linspace(em_wavelengths[0], em_wavelengths[-1], 128, dtype=np.float32)

                self.data_registry[f_name] = {
                    "ex_wavelengths": ex_wavelengths,
                    "em_wavelengths": em_wavelengths,
                    "sr_ex_waves": sr_ex_waves,
                    "sr_em_waves": sr_em_waves,
                    "lr_plot_data": lr_plot_data,
                    "lr_plot_data1": lr_plot_data1,
                    "sr_data": None
                }
                dropdown_list.append(f_name)
            except Exception as e:
                messagebox.showerror("Matrix Parse Error", f"File broken or illegal format [{f_name}]:\n{str(e)}")

        if dropdown_list:
            self.cb_file_selector.config(state="readonly", values=dropdown_list)
            self.cb_file_selector.current(0)
            self.active_key = dropdown_list[0]

            self.slider_ex.config(from_=0, to=127)
            self.slider_em.config(from_=0, to=127)

            self.btn_sr.config(state=tk.NORMAL)
            self.replotted_canvas()
            self.update_status(f"Batch files loaded: {len(dropdown_list)} text items ready for processing.")

    def on_dropdown_switch(self, event):
        self.active_key = self.cb_file_selector.get()
        cache = self.data_registry.get(self.active_key)

        if cache["sr_data"] is not None:
            self.btn_export.config(state=tk.NORMAL)
        else:
            self.btn_export.config(state=tk.DISABLED)

        self.replotted_canvas()
        self.update_status(f"Switched display target file to: {self.active_key}")

    def run_all_super_resolution(self):
        if self.model is None or not self.data_registry:
            return

        self.update_status("Processing batch deep learning forward pass...")
        self.root.config(cursor="watch")
        self.root.update()

        success_count = 0
        try:
            for f_name, cache in self.data_registry.items():
                lr_plot_data1 = cache["lr_plot_data1"]
                lr_tensor = torch.from_numpy(lr_plot_data1).unsqueeze(0).unsqueeze(0).to(self.device)
                H, W = lr_tensor.shape[2], lr_tensor.shape[3]

                pad_h = (4 - H % 4) % 4
                pad_w = (4 - W % 4) % 4
                import torch.nn.functional as F
                lr_padded = F.pad(lr_tensor, (0, pad_w, 0, pad_h), mode='replicate')

                with torch.no_grad():
                    sr_tensor_padded = self.model(lr_padded)

                sr_tensor = sr_tensor_padded[:, :, :H * 2, :W * 2]
                cache["sr_data"] = sr_tensor.cpu().squeeze().numpy()
                success_count += 1

            self.btn_export.config(state=tk.NORMAL)
            self.replotted_canvas()
            self.update_status(f"Batch super-resolution completed. Processed {success_count} entries.")
        except Exception as e:
            messagebox.showerror("Inference Error", f"Failure in network propagation:\n{str(e)}")
            self.update_status("Inference task broken.")
        finally:
            self.root.config(cursor="")

    def replotted_canvas(self):
        if not self.active_key:
            return

        cache = self.data_registry.get(self.active_key)
        if not cache:
            return

        lr_mat = cache["lr_plot_data"]
        sr_mat = cache["sr_data"]

        idx_ex = self.ex_slider_val.get()
        idx_em = self.em_slider_val.get()

        curr_ex_wave = np.linspace(cache["ex_wavelengths"][0], cache["ex_wavelengths"][-1], 128)[idx_ex]
        curr_em_wave = np.linspace(cache["em_wavelengths"][0], cache["em_wavelengths"][-1], 128)[idx_em]
        self.lbl_slider_ex.config(text=f"Excitation (EX) Wavelength: {curr_ex_wave:.1f} nm")
        self.lbl_slider_em.config(text=f"Emission (EM) Wavelength: {curr_em_wave:.1f} nm")

        for ax in self.axes.flatten():
            ax.clear()

        extent_lr = [cache["ex_wavelengths"][0], cache["ex_wavelengths"][-1], cache["em_wavelengths"][0],
                     cache["em_wavelengths"][-1]]
        extent_sr = [cache["sr_ex_waves"][0], cache["sr_ex_waves"][-1], cache["sr_em_waves"][0],
                     cache["sr_em_waves"][-1]]

        self.ax_lr.set_box_aspect(1)
        self.ax_lr.imshow(lr_mat, cmap=parula.parula_map, origin='lower', aspect='auto', extent=extent_lr,
                          interpolation='nearest')
        self.ax_lr.set_title(f"LR Input ({cache['ex_wavelengths'].shape[0]}×{cache['em_wavelengths'].shape[0]})",
                             fontsize=10, fontweight='bold', pad=6)
        self.ax_lr.axvline(x=curr_ex_wave, color='white', linestyle='--', alpha=0.6, linewidth=1)
        self.ax_lr.axhline(y=curr_em_wave, color='white', linestyle='--', alpha=0.6, linewidth=1)
        self.ax_lr.set_xlabel("Excitation Wavelength (nm)")
        self.ax_lr.set_ylabel("Emission Wavelength (nm)")

        self.ax_sr.set_box_aspect(1)
        if sr_mat is not None:
            self.ax_sr.imshow(sr_mat, cmap=parula.parula_map, origin='lower', aspect='auto', extent=extent_lr,
                              interpolation='nearest')
            self.ax_sr.set_title(f"SR Reconstruction ({sr_mat.shape[1]}×{sr_mat.shape[0]})", fontsize=10,
                                 fontweight='bold', pad=6)
            self.ax_sr.axvline(x=curr_ex_wave, color='white', linestyle='--', alpha=0.6, linewidth=1)
            self.ax_sr.axhline(y=curr_em_wave, color='white', linestyle='--', alpha=0.6, linewidth=1)
        else:
            self.ax_sr.text(0.5, 0.5, "Awaiting Batch Inference...", ha='center', va='center',
                            transform=self.ax_sr.transAxes)
            self.ax_sr.set_title("SR Output", fontsize=10, fontweight='bold', pad=6)
        self.ax_sr.set_xlabel("Excitation Wavelength (nm)")
        self.ax_sr.set_ylabel("Emission Wavelength (nm)")

        lr_profile_ex = lr_mat[:, idx_ex]
        lr_x_axis = np.linspace(cache["em_wavelengths"][0], cache["em_wavelengths"][-1], len(lr_profile_ex))

        self.ax_ex_slice.plot(lr_x_axis, lr_profile_ex, color='#475669', linewidth=1.5, drawstyle='steps-mid',
                              label='LR Input')

        if sr_mat is not None:
            idx_ex_sr = int(idx_ex * (sr_mat.shape[1] / lr_mat.shape[1]))
            if idx_ex_sr < sr_mat.shape[1]:
                sr_profile_ex = sr_mat[:, idx_ex_sr]
                sr_x_axis = np.linspace(cache["em_wavelengths"][0], cache["em_wavelengths"][-1], len(sr_profile_ex))
                self.ax_ex_slice.plot(sr_x_axis, sr_profile_ex, color='#20a0ff', linewidth=1.5, label='SR Output')

        self.ax_ex_slice.set_title(f"Emission Profile (at Ex = {curr_ex_wave:.1f} nm)", fontsize=9, fontweight='bold')
        self.ax_ex_slice.set_xlabel("Emission Wavelength (nm)")
        self.ax_ex_slice.set_ylabel("Fluorescence Intensity")
        self.ax_ex_slice.grid(True)
        self.ax_ex_slice.legend(frameon=True, fontsize=8)

        lr_profile_em = lr_mat[idx_em, :]
        lr_x_axis_em = np.linspace(cache["ex_wavelengths"][0], cache["ex_wavelengths"][-1], len(lr_profile_em))

        self.ax_em_slice.plot(lr_x_axis_em, lr_profile_em, color='#475669', linewidth=1.5, drawstyle='steps-mid',
                              label='LR Input')

        if sr_mat is not None:
            idx_em_sr = int(idx_em * (sr_mat.shape[0] / lr_mat.shape[0]))
            if idx_em_sr < sr_mat.shape[0]:
                sr_profile_em = sr_mat[idx_em_sr, :]
                sr_x_axis_em = np.linspace(cache["ex_wavelengths"][0], cache["ex_wavelengths"][-1], len(sr_profile_em))
                self.ax_em_slice.plot(sr_x_axis_em, sr_profile_em, color='#13ce66', linewidth=1.5, label='SR Output')

        self.ax_em_slice.set_title(f"Excitation Profile (at Em = {curr_em_wave:.1f} nm)", fontsize=9, fontweight='bold')
        self.ax_em_slice.set_xlabel("Excitation Wavelength (nm)")
        self.ax_em_slice.set_ylabel("Fluorescence Intensity")
        self.ax_em_slice.grid(True)
        self.ax_em_slice.legend(frameon=True, fontsize=8)

        self.canvas.draw()

    def export_to_txt(self):
        if not self.active_key:
            return
        cache = self.data_registry.get(self.active_key)
        if cache["sr_data"] is None:
            messagebox.showerror("Export Error", "No high-resolution result available to save.")
            return

        default_name = f"SR_{os.path.splitext(self.active_key)[0]}.txt"
        file_path = filedialog.asksaveasfilename(title="Save Super-Resolved Matrix as Text", initialfile=default_name,
                                                 defaultextension=".txt", filetypes=[("Text Data Files", "*.txt")])
        if not file_path:
            return

        try:
            self.update_status("Formatting and exporting high-resolution text data...")
            sr_mat = cache["sr_data"]
            sr_ex = cache["sr_ex_waves"]
            sr_em = cache["sr_em_waves"]

            output_matrix = np.zeros((sr_mat.shape[0] + 1, sr_mat.shape[1] + 1), dtype=np.float64)
            output_matrix[0, 1:] = sr_ex
            output_matrix[1:, 0] = sr_em
            output_matrix[1:, 1:] = sr_mat

            np.savetxt(file_path, output_matrix, fmt='%.6f', delimiter='\t')
            messagebox.showinfo("Export Success", f"Reconstructed text file saved successfully:\n{file_path}")
            self.update_status("Text data exported successfully.")
        except Exception as e:
            messagebox.showerror("File IO Error", f"Unable to generate text file:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = FluorescenceSRGUI(root)
    root.mainloop()