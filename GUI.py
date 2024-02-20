import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import font
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import os
import time
import spiceypy

# dont need this, only for debug
import matplotlib.pyplot as plt

from Util import *
from Renderer import ViewRenderer
from SPICEHandler import SpiceDataHandler
from DataDownloader import PerijoveDataDownloader

### RESOLUTION IS FIXED TO 1600 x 900p



class GUIObject:
    def __init__(self, renderer=None, spice_data_handler=None, data_downloader=None):
        self.data_path = './Data'
        self.selected_perijove_path = None
        self.progress_bar_window = None
        self.download_progress_bar_window = None
        self.loaded_image_properties = {}
        self.currently_active_loaded_image = None
        self.loaded_images_stack = []

        self.enable_render = False

        self.root = tk.Tk()
            
        self.root.geometry("{}x{}".format(1600, 900))
        self.root.resizable(False, False)
        self.root.title('JunoCam Raw Image Viewer')

        self.renderer = renderer
        self.spice_data_handler = spice_data_handler
        self.data_downloader = data_downloader
        if self.renderer is not None:
            assert self.renderer.spice_data_handler == self.spice_data_handler

        self.set_tk_control_variables()
        self.pack_widgets()
        self.update_cam_view_controls()


    def set_tk_control_variables(self):
        self.view_cam_latitude_var = tk.DoubleVar(self.root, 0)
        self.view_cam_latitude_var_entry = tk.StringVar(self.root, '0.0')
        self.view_cam_longitude_var = tk.DoubleVar(self.root, 0)
        self.view_cam_longitude_var_entry = tk.StringVar(self.root, '0.0')
        self.view_cam_height_var = tk.DoubleVar(self.root, 0)
        self.view_cam_height_var_entry = tk.StringVar(self.root, '0.0')
        self.view_cam_height_log_var = tk.DoubleVar(self.root, np.log(self.view_cam_height_var.get() + 1.0) - 2.3978952728)
        self.view_cam_roll_var = tk.DoubleVar(self.root, 0)
        self.roll_lock_var = tk.IntVar(self.root, 0)
        self.atm_at_horizon_selection_var = tk.IntVar(self.root, 1)
        self.atm_selection_var = tk.IntVar(self.root, 1)
        self.atm_r_var_entry = tk.StringVar(self.root, '138')
        self.atm_g_var_entry = tk.StringVar(self.root, '107')
        self.atm_b_var_entry = tk.StringVar(self.root, '45')
        self.atm_color = [138, 107, 45]
        self.renderer.set_atm_color(self.atm_color)
        self.render_res_x_var = tk.StringVar(self.root, '1024')
        self.render_res_y_var = tk.StringVar(self.root, '1024')
        self.supersampling_var = tk.IntVar(self.root, 1)
        self.cam_focal_length_text_var = tk.StringVar(self.root, ' Camera focal length: 2.000')
        self.main_view_focal_length_text_var = tk.StringVar(self.root, ' Main view focal length: 2.000')
        self.blending_horizon_var = tk.DoubleVar(self.root, 0.3)
        self.blending_edge_var = tk.DoubleVar(self.root, 0.5)


    def get_render_res(self):
        x, y = self.render_res_x_var.get(), self.render_res_y_var.get()
        x = 1 if x == '' else int(x)
        y = 1 if y == '' else int(y)
        return (x, y)


    def destroy_yourself(self):
        self.root.destroy()


    def set_data_downloader(self, data_downloader):
        self.data_downloader = data_downloader


    def set_renderer(self, renderer):
        self.renderer = renderer
        assert self.renderer.spice_data_handler == self.spice_data_handler

    def set_spice_data_handler(self, spice_data_handler):
        self.spice_data_handler = spice_data_handler
        if self.renderer is not None:
            assert self.renderer.spice_data_handler == self.spice_data_handler

    def display_insufficient_data(self):
        messagebox.showerror(title='Insufficient Data', message='Not all SPICE kernels required to load this image are available.')


    def show_could_not_download_error(self):
        messagebox.showerror(title='Download error', message='The download of some file failed after five retries. Check your internet connection or maybe just try again.')


    def load_image(self, img_name):
        assert self.selected_perijove_path is not None
        self.enable_render = False
        self.start_progress_bar(text='', title_text='Preprossesing Image')

        try:
            self.renderer.load_image(self.selected_perijove_path, img_name)
        except spiceypy.utils.exceptions.SpiceSPKINSUFFDATA:
            self.close_progress_bar()
            self.display_insufficient_data()

        self.loaded_image_properties[img_name] = {'framelet_list': [True for i in range(self.renderer.get_num_framelets(img_name))],
                                                  'time_offset': 0.0062,
                                                  'interframe_delay': 0.0013}
        self.time_offset_slider.set(62)
        self.time_offset_scale_change(None)
        self.interframe_delay_slider.set(1.3)
        self.interframe_delay_scale_change(None)

        self.enable_render = True
        self.render_views()

        self.close_progress_bar()


    def start_download_progress_bar(self, perijove_number, exit_command=None):
        self.download_progress_bar_window = tk.Toplevel(self.root)
        self.download_progress_bar_window.is_alive = True

        def do_on_exit():
            self.download_progress_bar_window.is_alive = False
            exit_command()
            self.download_progress_bar_window.destroy()
            self.data_downloader.cleanup_temp_folder()

        if exit_command is not None:
            self.download_progress_bar_window.protocol("WM_DELETE_WINDOW", do_on_exit)
            

        self.download_progress_bar_window.geometry('550x210')
        self.download_progress_bar_window.geometry('+700+400')
        self.download_progress_bar_window.title('Downloading PJ{}'.format(perijove_number))
        self.download_progress_bar_window.resizable(False, False)
        self.download_progress_bar_window.grab_set()

        self.download_progress_bar_window.prog_bars = []
        self.download_progress_bar_window.labels = []

        tk.Label(self.download_progress_bar_window, width=3).grid(row=0, column=0)
        for i in range(3):
            tk.Label(self.download_progress_bar_window, height=1).grid(row=0 + i * 3, column=1)
            new_prog_bar = ttk.Progressbar(self.download_progress_bar_window, orient='horizontal', mode='determinate', length=500)
            new_prog_bar.grid(row=1 + i * 3, column=1, sticky='w')
            self.download_progress_bar_window.prog_bars.append(new_prog_bar)

            new_prog_bar_text = tk.Label(self.download_progress_bar_window, height=1, anchor='w')
            new_prog_bar_text.grid(row=2 + i * 3, column=1, sticky='w')
            self.download_progress_bar_window.labels.append(new_prog_bar_text)

        self.root.update()


    def update_download_progress_bar(self, prog_bar_index, delta, text=None):
        if self.download_progress_bar_window is None or not self.download_progress_bar_window.is_alive:
            return 
        prog_bar = self.download_progress_bar_window.prog_bars[prog_bar_index]
        label = self.download_progress_bar_window.labels[prog_bar_index]
        prog_bar.step(delta)
        if text is not None:
            label.config(text=text)
        self.root.update()


    def close_download_progress_bar(self):
        if self.download_progress_bar_window is None or not self.download_progress_bar_window.is_alive:
            return
        self.download_progress_bar_window.destroy()
        self.download_progress_bar_window = None
        self.root.update()


    def start_progress_bar(self, text='', title_text=''):
        self.progress_bar_window = tk.Toplevel(self.root)
        self.progress_bar_window.is_alive = True

        def callback():
            self.progress_bar_window.is_alive = False
            self.progress_bar_window.destroy()

        self.progress_bar_window.protocol("WM_DELETE_WINDOW", callback)
        self.progress_bar_window.geometry('350x120')
        self.progress_bar_window.geometry('+700+400')
        self.progress_bar_window.title(title_text)
        self.progress_bar_window.resizable(False, False)
        self.progress_bar_window.grab_set()

        tk.Label(self.progress_bar_window, height=2).pack()
        self.progress_bar_window.prog_bar = ttk.Progressbar(self.progress_bar_window, orient='horizontal', mode='determinate', length=300)
        self.progress_bar_window.prog_bar.pack()
        self.progress_bar_window.prog_bar_text = tk.Label(self.progress_bar_window, text=text, height=2)
        self.progress_bar_window.prog_bar_text.pack()
        self.root.update()


    def update_progress_bar(self, delta, text=None):
        if self.progress_bar_window is None:
            return
        if not self.progress_bar_window.is_alive:
            return 1
        self.progress_bar_window.prog_bar.step(delta)
        if text is not None:
            self.progress_bar_window.prog_bar_text.config(text=text)
        self.root.update()

    def close_progress_bar(self):
        if self.progress_bar_window is None or not self.progress_bar_window.is_alive:
            return
        self.progress_bar_window.destroy()
        self.progress_bar_window = None
        self.root.update()


    def load_perijove_data(self, perijove_name):
        perijove_dir = os.path.join(self.data_path, perijove_name)
        self.selected_perijove_path = perijove_dir
        image_names = [x.split('.')[0] for x in os.listdir(os.path.join(perijove_dir, 'images_info'))]

        self.renderer.clear_loaded_images()
        self.render_views()

        self.start_progress_bar(text='Loading Images..', title_text='Loading Images')

        self.spice_data_handler.load_kernels(self.data_path, perijove_name)

        box_height = 96
        self.image_list_canvas.delete("all")
        for canvas in [self.image_list_canvas, self.main_view_canvas, self.cam_view_canvas]:
            canvas.configure(bd=-3)
            canvas.update()
        self.image_list_canvas.thumbnail_list = []
        self.image_list_canvas.file_dict = {}
        self.loaded_images_stack = []
        self.currently_active_loaded_image = None
        self.image_list_canvas.loaded_rects = []
        self.image_list_canvas.activated_rect = None
        for k, image_name in enumerate(image_names):
            image_container_box = self.image_list_canvas.create_rectangle(0, k * box_height + 1, 255, (k+1) * box_height - 1, fill='gray64', tag='not_active')
            self.image_list_canvas.file_dict[image_container_box] = image_name
            self.image_list_canvas.create_text(6, (k + 0.5) * box_height, anchor='w', text=image_name)
            img = Image.open(os.path.join(perijove_dir, 'images', image_name + '.png'))
            img.thumbnail((box_height - 10, box_height - 10))

            photo_image = ImageTk.PhotoImage(img)
            self.image_list_canvas.thumbnail_list.append(photo_image)
            self.image_list_canvas.create_image(210, (k + 0.5) * box_height, anchor='c', image=photo_image)

            error_code = self.update_progress_bar(100 / len(image_names), text='Loading Image: ' + image_name)
            if error_code == 1:
                break


        def mouse_motion(event):
            a, b = event.x, self.image_list_canvas.canvasy(event.y)
            last_active_ids = [x for x in self.image_list_canvas.find_withtag('last_active') if self.image_list_canvas.type(x) == "rectangle"]
            overlapping_ids = [x for x in self.image_list_canvas.find_overlapping(a, b, a, b) if self.image_list_canvas.type(x) == "rectangle"]

            for rect_id in last_active_ids:
                if rect_id not in overlapping_ids:
                    if rect_id in self.image_list_canvas.loaded_rects:
                        self.image_list_canvas.itemconfigure(rect_id, fill='gray28', tag='not_active')
                    else:
                        self.image_list_canvas.itemconfigure(rect_id, fill='gray64', tag='not_active')
            
            for rect_id in overlapping_ids:
                cur_fill = self.image_list_canvas.itemcget(rect_id, 'fill')
                if cur_fill == 'gray64' or cur_fill == 'gray28':
                    self.image_list_canvas.itemconfigure(rect_id, fill='gray42', tag='last_active')

        def leave(event):
            last_active_ids = [x for x in self.image_list_canvas.find_withtag('last_active') if self.image_list_canvas.type(x) == "rectangle"]
            for rect_id in last_active_ids:
                if rect_id in self.image_list_canvas.loaded_rects:
                    self.image_list_canvas.itemconfigure(rect_id, fill='gray28', tag='not_active')
                else:
                    self.image_list_canvas.itemconfigure(rect_id, fill='gray64', tag='not_active')
        
        def double_click(event):
            a, b = event.x, self.image_list_canvas.canvasy(event.y)
            overlapping_ids = [x for x in self.image_list_canvas.find_overlapping(a, b, a, b) if self.image_list_canvas.type(x) == "rectangle"]
            if len(overlapping_ids) == 0:
                return
            rect_id = overlapping_ids[0]
            name = self.image_list_canvas.file_dict[rect_id]
            self.image_list_canvas.itemconfigure(rect_id, fill='gray42', tag='last_active')
            if name in self.loaded_images_stack:
                self.image_list_canvas.itemconfigure(rect_id, width=1, outline='black')
                x0, y0, x1, y1 = self.image_list_canvas.coords(rect_id)
                self.image_list_canvas.coords(rect_id, x0, y0 - 1, x1, y1 + 1)
                self.image_list_canvas.loaded_rects.remove(rect_id)
                self.loaded_images_stack.remove(name)
                if rect_id == self.image_list_canvas.activated_rect:
                    if len(self.loaded_images_stack) > 0:
                        self.image_list_canvas.activated_rect = self.image_list_canvas.loaded_rects[-1]
                        self.currently_active_loaded_image = self.loaded_images_stack[-1]
                        self.image_list_canvas.itemconfigure(self.image_list_canvas.activated_rect, width=2, outline='red')
                    else:
                        self.image_list_canvas.activated_rect = None
                        self.currently_active_loaded_image = None
                self.unload_image(name)
            else:
                self.image_list_canvas.itemconfigure(rect_id, width=2, outline='red')
                x0, y0, x1, y1 = self.image_list_canvas.coords(rect_id)
                self.image_list_canvas.coords(rect_id, x0, y0 + 1, x1, y1 - 1)
                self.image_list_canvas.loaded_rects.append(rect_id)
                if self.image_list_canvas.activated_rect is not None:
                    self.image_list_canvas.itemconfigure(self.image_list_canvas.activated_rect, outline='black')
                self.image_list_canvas.activated_rect = rect_id
                self.currently_active_loaded_image = name
                self.loaded_images_stack.append(name)
                self.load_image(name)

        def click(event):
            a, b = event.x, self.image_list_canvas.canvasy(event.y)
            overlapping_ids = [x for x in self.image_list_canvas.find_overlapping(a, b, a, b) if self.image_list_canvas.type(x) == "rectangle"]
            if len(overlapping_ids) == 0:
                return
            rect_id = overlapping_ids[0]
            name = self.image_list_canvas.file_dict[rect_id]
            if name in self.loaded_images_stack:
                if self.image_list_canvas.activated_rect is not None:
                    self.image_list_canvas.itemconfigure(self.image_list_canvas.activated_rect, outline='black')
                self.image_list_canvas.activated_rect = rect_id
                self.currently_active_loaded_image = name
                self.set_active_loaded_image(name)
                self.image_list_canvas.itemconfigure(rect_id, width=2, outline='red')


        def scroll_canvas(event):
            self.image_list_canvas.yview(tk.SCROLL, -event.delta // 120, 'unit')


        self.image_list_canvas.bind('<Motion>', mouse_motion)
        self.image_list_canvas.bind('<Double-Button-1>', double_click)
        self.image_list_canvas.bind('<Button-1>', click)
        self.image_list_canvas.bind('<MouseWheel>', scroll_canvas)
        self.image_list_canvas.bind('<Leave>', leave)

        self.image_list_canvas.configure(scrollregion=self.image_list_canvas.bbox('all'))
        for canvas in [self.image_list_canvas, self.main_view_canvas, self.cam_view_canvas]:
            canvas.configure(bd=-2)
            canvas.update()

        self.close_progress_bar()


    def show_insufficient_kernel_data_warning(self):
        message = """No spacecraft kernels for the image timeframes were found in the NAIF data repository. Without this information, raw images cannot be projected."""
        messagebox.showwarning(title='Insufficient data', message=message)


    def set_active_loaded_image(self, name):
        time_offset = self.loaded_image_properties[name]['time_offset']
        interframe_delay = self.loaded_image_properties[name]['interframe_delay']

        self.enable_render = False
        self.time_offset_slider.set(time_offset * 1e+3)
        self.time_offset_scale_change(None)
        self.interframe_delay_slider.set(interframe_delay * 1e+3)
        self.interframe_delay_scale_change(None)
        self.enable_render = True


    def unload_image(self, name):
        self.loaded_image_properties.pop(name)
        self.renderer.unload_image(name)

        self.enable_render = False
        self.time_offset_slider.set(62)
        self.time_offset_scale_change(None)
        self.interframe_delay_slider.set(1.3)
        self.interframe_delay_scale_change(None)
        self.enable_render = True

        self.render_views()


    def select_perijove(self):
        window = tk.Toplevel(self.root)
        window.geometry('+750+400')
        window.title('Select Approach')
        window.resizable(False, False)
        window.grab_set()

        perijove_list = tk.Listbox(window, width=10, activestyle='none')

        def update_data_list():
            data_list = [' '*5 + x for x in os.listdir(self.data_path) if x[:2] == 'PJ']
            perijove_list.delete(0, perijove_list.size())
            perijove_list.insert(0, *data_list)
            self.root.update()

        update_data_list()
        perijove_list.grid(row=0, column=0, sticky='w')

        def perijove_select():
            element = perijove_list.curselection()
            if len(element) == 0:
                return
            element = perijove_list.get(element[0]).strip()
            window.destroy()
            self.load_perijove_data(element)

        def download_new():
            new_window = tk.Toplevel(window)
            new_window.geometry('+790+500')
            new_window.grab_set()
            tk.Label(new_window, text='Enter number of Perijove to download:  ').grid(row=0, column=0)
            entry = tk.Entry(new_window, width=9)
            entry.grid(row=0, column=1)

            def submit():
                text = entry.get()
                if not text.isdigit():
                    messagebox.showerror(title='Error', message='Not a valid Perijove number!')
                    new_window.destroy()
                    window.lift()
                    return

                perijove_number = int(text)
                perijove_range = self.data_downloader.get_perijove_range()
                if perijove_range is None:
                    messagebox.showerror(title='Failed Connection', message='Could not establish connection to NASA website.')
                if perijove_number < perijove_range[0] or perijove_number > perijove_range[1]:
                    messagebox.showerror(title='Error', message='Only Perijove numbers between {} and {} are currently available'.format(*perijove_range))
                    new_window.destroy()
                    window.lift()
                    return

                new_window.destroy()
                window.lift()
                self.spice_data_handler.clear_kernels()
                error_code = self.data_downloader.download(perijove_number)
                if error_code == 1:
                    self.show_insufficient_kernel_data_warning()
                update_data_list()

            tk.Label(new_window, width=1).grid(row=0, column=2)
            tk.Button(new_window, text='Submit', command=submit, width=10).grid(row=0, column=3)


        select_button = tk.Button(window, text='Select', width=20, command=perijove_select)
        select_button.grid(row=1, column=0)

        download_new_button = tk.Button(window, text='Download New', width=20, command=download_new)
        download_new_button.grid(row=1, column=1)


    def get_main_view_camera_mat(self):
        shiftx_dir = np.cross(self.main_view_camera_pos, np.array((0.0, 0.0, 1.0)))
        shift_x_norm = -np.linalg.norm(shiftx_dir)
        if shift_x_norm == 0:
            shiftx_dir = np.array((1.0, 0.0, 0.0))
        else:
            shiftx_dir /= shift_x_norm

        camera_center_dist = np.linalg.norm(self.main_view_camera_pos)
        shift_y_dir = -np.cross(shiftx_dir, self.main_view_camera_pos) / camera_center_dist

        camera_mat = np.array([shiftx_dir, shift_y_dir, -self.main_view_camera_pos * self.main_view_focal_length / camera_center_dist])
        return camera_mat


    def render_views(self):
        self.render_main_view()
        self.render_cam_view()


    def get_wf_render_mode(self, disable_cam=False):
        camera_selection = 0 if disable_cam else self.camera_selection_var.get()
        wf_render_mode = (self.ellipsoid_selection_var.get(),
                          self.mesh_selection_var.get(),
                          self.mesh_on_top_selection_var.get(),
                          camera_selection,
                          self.atm_selection_var.get(),
                          self.atm_at_horizon_selection_var.get())
        return wf_render_mode


    def get_blending_params(self):
        return (self.blending_horizon_var.get(), min(self.blending_edge_var.get(), 0.999))


    def render_main_view(self):
        if not self.enable_render:
            return
        camera_mat = self.get_main_view_camera_mat()
        wf_render_mode = self.get_wf_render_mode()
        extra_info = [self.cam_view_camera_pos, self.cam_view_camera_mat, self.main_view_focal_length, self.cam_view_focal_length]
        view_ar = self.renderer.render_view(self.main_view_camera_pos, camera_mat, 0, self.loaded_image_properties,
            self.get_blending_params(), render_res=self.get_render_res(), wf_render_mode=wf_render_mode, cam_view_info=extra_info)

        if not hasattr(self.main_view_canvas, 'view_image'):
            img = Image.fromarray(view_ar[::-1, :])
            photo_image = ImageTk.PhotoImage(img)
            self.main_view_canvas.view_image = photo_image
            self.main_view_canvas.delete('all')
            self.main_view_canvas.create_image(450, 450, image=photo_image)
        else:
            img = Image.fromarray(view_ar[::-1, :])
            self.main_view_canvas.view_image.paste(img)


    def render_cam_view(self):
        if not self.enable_render:
            return
        cam_view_cam_mat = self.cam_view_camera_mat.copy()
        cam_view_cam_mat[2] *= self.cam_view_focal_length
        wf_render_mode = self.get_wf_render_mode(disable_cam=True)
        view_ar = self.renderer.render_view(self.cam_view_camera_pos, cam_view_cam_mat, 1, self.loaded_image_properties, self.get_blending_params(),
            render_res=self.get_render_res(), wf_render_mode=wf_render_mode)

        if not hasattr(self.cam_view_canvas, 'view_image'):
            img = Image.fromarray(view_ar[::-1, :])
            photo_image = ImageTk.PhotoImage(img)
            self.cam_view_canvas.view_image = photo_image
            self.cam_view_canvas.delete('all')
            self.cam_view_canvas.create_image(212, 212, image=photo_image)
        else:
            img = Image.fromarray(view_ar[::-1, :])
            self.cam_view_canvas.view_image.paste(img)


    
    def init_main_view_camera(self):
        self.main_view_camera_pos = np.array((500000., 0., 0.))
        self.main_view_focal_length = 2.0


    def init_cam_view_camera(self):
        self.cam_view_camera_pos = np.array((80000., 30000., 30000.))
        self.cam_view_camera_mat = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        self.cam_view_focal_length = 2.0


    def bind_cam_view_controls(self):
        self.cam_view_keys = {'button_1': False, 'last_button_pos': None}

        def click(event):
            self.cam_view_keys['button_1'] = True
            self.cam_view_keys['last_button_pos'] = np.array((event.x, event.y))

        def motion(event):
            if self.cam_view_keys['button_1'] and self.cam_view_keys['last_button_pos'] is not None:
                pos = np.array((event.x, event.y))
                delta = pos - self.cam_view_keys['last_button_pos']
                self.cam_view_keys['last_button_pos'] = pos

                fixed_roll = None
                if self.roll_lock_var.get() == 1:
                    fixed_roll = self.view_cam_roll_var.get()
                self.cam_view_camera_mat = move_cam_view(self.cam_view_camera_pos, self.cam_view_camera_mat,
                    delta * 0.003 / self.cam_view_focal_length, fixed_roll)
                self.update_cam_view_controls()
                self.render_views()

        def click_release(event):
            self.cam_view_keys['button_1'] = False
            self.cam_view_keys['last_button_pos'] = None

        def scroll(event):
            if event.state == 8:
                fixed_roll = None
                if self.roll_lock_var.get() == 1:
                    fixed_roll = self.view_cam_roll_var.get()
                self.cam_view_camera_pos, self.cam_view_camera_mat = scroll_view_cam_view(self.cam_view_camera_pos, self.cam_view_camera_mat,
                    event.delta * 0.005, fixed_roll)
                self.update_cam_view_controls()
                self.render_views()
            elif event.state == 9:
                self.cam_view_focal_length *= np.exp(0.0005 * event.delta)
                self.cam_focal_length_text_var.set(' Camera focal length: {:.3f}'.format(self.cam_view_focal_length))
                self.render_views()

        def leave(event):
            self.cam_view_keys['button_1'] = False
            self.cam_view_keys['last_button_pos'] = None

        self.cam_view_canvas.bind('<Button-1>', click)
        self.cam_view_canvas.bind('<Motion>', motion)
        self.cam_view_canvas.bind('<ButtonRelease-1>', click_release)
        self.cam_view_canvas.bind('<MouseWheel>', scroll)
        self.cam_view_canvas.bind('<Leave>', leave)


    def bind_main_view_controls(self):
        self.main_view_keys = {'button_1': False, 'last_button_pos': None}

        def click(event):
            self.main_view_keys['button_1'] = True
            self.main_view_keys['last_button_pos'] = np.array((event.x, event.y))

        def motion(event):
            if self.main_view_keys['button_1'] and self.main_view_keys['last_button_pos'] is not None:
                pos = np.array((event.x, event.y))
                delta = pos - self.main_view_keys['last_button_pos']
                self.main_view_keys['last_button_pos'] = pos
                self.main_view_camera_pos = move_main_view(self.main_view_camera_pos, delta * 0.006 / self.main_view_focal_length)
                self.render_main_view()

        def click_release(event):
            self.main_view_keys['button_1'] = False
            self.main_view_keys['last_button_pos'] = None

        def scroll(event):
            if event.state == 8:
                self.main_view_camera_pos = scroll_view_to_surface(self.main_view_camera_pos, event.delta * 0.001)
                self.render_main_view()
            elif event.state == 9:
                self.main_view_focal_length *= np.exp(0.0005 * event.delta)
                self.main_view_focal_length_text_var.set(' Main view focal length: {:.3f}'.format(self.main_view_focal_length))
                self.render_main_view()

        def leave(event):
            self.main_view_keys['button_1'] = False
            self.main_view_keys['last_button_pos'] = None

        self.main_view_canvas.bind('<Button-1>', click)
        self.main_view_canvas.bind('<Motion>', motion)
        self.main_view_canvas.bind('<ButtonRelease-1>', click_release)
        self.main_view_canvas.bind('<MouseWheel>', scroll)
        self.main_view_canvas.bind('<Leave>', leave)
        

    def time_offset_scale_change(self, event):
        if self.currently_active_loaded_image is None:
            return
        val = self.time_offset_slider.get() * 1e-3
        self.loaded_image_properties[self.currently_active_loaded_image]['time_offset'] = val
        if self.renderer is not None:
            self.render_views()


    def surface_height_scale_change(self, event):
        val = self.surface_height_slider.get()
        if self.renderer is not None:
            self.renderer.set_height_offset(val)
            self.render_views()


    def interframe_delay_scale_change(self, event):
        if self.currently_active_loaded_image is None:
            return
        val = self.interframe_delay_slider.get() * 1e-3
        self.loaded_image_properties[self.currently_active_loaded_image]['interframe_delay'] = val
        if self.renderer is not None:
            self.render_views()


    def reset_atm_color(self):
        atm_color = [138, 107, 45]
        entry_vars = [self.atm_r_var_entry, self.atm_g_var_entry, self.atm_b_var_entry]
        for k, entry_var in enumerate(entry_vars):
            entry_var.set(str(atm_color[k]))

        self.atm_color = atm_color
        self.renderer.set_atm_color(self.atm_color)


    def reset_controls(self):
        self.reset_atm_color()
        self.reset_time_offset_slider()
        self.reset_interframe_delay_slider()
        self.reset_surface_height_slider()
        

    def reset_time_offset_slider(self):
        self.time_offset_slider.set(62)
        self.time_offset_scale_change(None)


    def reset_interframe_delay_slider(self):
        self.interframe_delay_slider.set(1.3)
        self.interframe_delay_scale_change(None)

    def reset_surface_height_slider(self):
        self.surface_height_slider.set(0)
        self.surface_height_scale_change(None)


    def select_framelets(self):
        if self.currently_active_loaded_image is None:
            return

        num_framelets = self.renderer.get_num_framelets(self.currently_active_loaded_image)

        window = tk.Toplevel(self.root)
        window.geometry('200x350')
        window.geometry('+700+300')
        window.title('')
        window.resizable(False, False)
        window.grab_set()

        tk.Label(window, text='Select the framelets to display:').place(x=4, y=6)

        list_checkbox = ScrolledText(window, width=10, height=18, cursor='arrow')
        list_checkbox['state'] = 'disabled'
        list_checkbox.place(x=4, y=32)

        def scroll_list(event):
            list_checkbox.yview(tk.SCROLL, -event.delta // 3, 'pixels')

        selected_framelets_list = self.loaded_image_properties[self.currently_active_loaded_image]['framelet_list']
        checkbutton_var_list = [tk.IntVar(window, int(x)) for x in selected_framelets_list]
        for i in range(num_framelets):
            cb = tk.Checkbutton(list_checkbox, text='Nr. {:02d}'.format(i+1), variable=checkbutton_var_list[i], bg='white', anchor='w', onvalue=1, offvalue=0)
            cb.var = checkbutton_var_list[i]
            cb.bind('<MouseWheel>', scroll_list)
            list_checkbox.window_create('end', window=cb)

        def select_all(val):
            for i in range(num_framelets):
                checkbutton_var_list[i].set(val)

        tk.Button(window, text='Select All', command=lambda:select_all(1)).place(x=105, y=32, width=95, height=24)
        tk.Button(window, text='Select None', command=lambda:select_all(0)).place(x=105, y=56, width=95, height=24)

        def confirm():
            self.loaded_image_properties[self.currently_active_loaded_image]['framelet_list'] = [x.get() == 1 for x in checkbutton_var_list]
            self.render_views()
            window.destroy()

        confirm_button = tk.Button(window, text='Confirm', command=confirm)
        confirm_button.place(x=0, y=326, width=200, height=24)


    def set_atm_color_controls(self):
        tk.Label(self.main_control_frame, text='  Atmosphere color:', justify=tk.LEFT, anchor='w').place(x=208, y=411, width=206, height=20)
        tk.Label(self.main_control_frame, text='  R:', justify=tk.LEFT, anchor='w').place(x=20 + 208, y=431, width=48, height=20)
        tk.Label(self.main_control_frame, text='  G:', justify=tk.LEFT, anchor='w').place(x=68 + 208, y=431, width=48, height=20)
        tk.Label(self.main_control_frame, text='  B:', justify=tk.LEFT, anchor='w').place(x=116 + 208, y=431, width=68, height=20)

        self.atm_r_entry = tk.Entry(self.main_control_frame, width=4, justify=tk.RIGHT, textvariable=self.atm_r_var_entry)
        self.atm_g_entry = tk.Entry(self.main_control_frame, width=4, justify=tk.RIGHT, textvariable=self.atm_g_var_entry)
        self.atm_b_entry = tk.Entry(self.main_control_frame, width=4, justify=tk.RIGHT, textvariable=self.atm_b_var_entry)
        self.atm_r_entry.place(x=42 + 208, y=431)
        self.atm_g_entry.place(x=90 + 208, y=431)
        self.atm_b_entry.place(x=138 + 208, y=431)

        def update_atm_color(*kaka):
            entry_vars = [self.atm_r_var_entry, self.atm_g_var_entry, self.atm_b_var_entry]
            for k, entry_var in enumerate(entry_vars):
                entry_string = entry_var.get()
                if entry_string == '':
                    self.atm_color[k] = 0
                    continue
                if not entry_string.isdigit():
                    entry_string = str(self.atm_color[k])
                val = int(entry_string)
                val = min(max(val, 0), 255)
                entry_var.set(str(val))
                self.atm_color[k] = val
            self.renderer.set_atm_color(self.atm_color)
            self.render_views()

        self.atm_r_var_entry.trace("w", update_atm_color)
        self.atm_g_var_entry.trace("w", update_atm_color)
        self.atm_b_var_entry.trace("w", update_atm_color)


    def export_main_view(self):
        if not self.enable_render:
            return

        camera_mat = self.get_main_view_camera_mat()
        wf_render_mode = self.get_wf_render_mode()
        extra_info = [self.cam_view_camera_pos, self.cam_view_camera_mat, self.main_view_focal_length, self.cam_view_focal_length]
        view_ar = self.renderer.render_view(self.main_view_camera_pos, camera_mat, 0, self.loaded_image_properties,
            self.get_blending_params(), render_res=self.get_render_res(), wf_render_mode=wf_render_mode, cam_view_info=extra_info)
        saving_path = filedialog.asksaveasfilename(filetypes=[('PNG', '*.png')])
        if saving_path == '':
            return
        if not saving_path[-4] == '.png':
            saving_path = saving_path + '.png'
        self.start_progress_bar(text='Saving..')
        Image.fromarray(view_ar[::-1, :]).save(saving_path)
        self.update_progress_bar(99.99)
        self.close_progress_bar()


    def render_final_view(self):
        if not self.enable_render:
            return
        cam_view_cam_mat = self.cam_view_camera_mat.copy()
        cam_view_cam_mat[2] *= self.cam_view_focal_length
        wf_render_mode = self.get_wf_render_mode(disable_cam=True)
        out_ar = self.renderer.render_final_view(self.cam_view_camera_pos, cam_view_cam_mat, self.loaded_image_properties, self.get_blending_params(),
            self.get_render_res(), wf_render_mode=wf_render_mode, super_sampling=(self.supersampling_var.get() == 1))

        saving_path = filedialog.asksaveasfilename(filetypes=[('PNG', '*.png')])
        if saving_path == '':
            return
        if not saving_path[-4] == '.png':
            saving_path = saving_path + '.png'
        self.start_progress_bar(text='Saving..')
        Image.fromarray(out_ar[::-1, :]).save(saving_path)
        self.update_progress_bar(99.99)
        self.close_progress_bar()
        



    def pack_main_control_frame(self):
        self.time_offset_slider = tk.Scale(self.main_control_frame, from_=-50, to=150, orient='horizontal',
                                           resolution=-1, label='Imaging time offset in ms:', width=13, length=200, command=self.time_offset_scale_change)
        self.time_offset_slider.place(x=0, y=0, width=206)

        self.interframe_delay_slider = tk.Scale(self.main_control_frame, from_=-3, to=8, orient='horizontal',
                                           resolution=-1, label='Interframe delay offset in ms:', width=13, length=200, command=self.interframe_delay_scale_change)
        self.interframe_delay_slider.place(x=0, y=70, width=206)


        self.surface_height_slider = tk.Scale(self.main_control_frame, from_=-500, to=500, orient='horizontal',
                                          resolution=-1, label='Surface height offset in km:', width=13, length=200, command=self.surface_height_scale_change)
        self.surface_height_slider.place(x=0, y=140, width=206)


        def render_views_scale(event):
            self.render_views()

        self.blending_slider1 = tk.Scale(self.main_control_frame, from_=0, to=1, orient='horizontal', var=self.blending_horizon_var,
                                          resolution=-1, label='Blending Hardness:', width=13, length=200, command=render_views_scale)
        self.blending_slider1.place(x=0, y=330, width=206)

        self.blending_slider2 = tk.Scale(self.main_control_frame, from_=0, to=1, orient='horizontal', var=self.blending_edge_var,
                                          resolution=-1, label='Blending at Edges:', width=13, length=200, command=render_views_scale)
        self.blending_slider2.place(x=0, y=385, width=206)


        self.select_framelets_button = tk.Button(self.main_control_frame, text='Select framelet subset', command=self.select_framelets)
        self.select_framelets_button.place(x=0, y=210, width=206, height=23)

        self.ellipsoid_selection_var = tk.IntVar(self.main_control_frame, 1)
        self.mesh_selection_var = tk.IntVar(self.main_control_frame, 1)
        self.mesh_on_top_selection_var = tk.IntVar(self.main_control_frame, 0)
        self.camera_selection_var = tk.IntVar(self.main_control_frame, 1)

        self.select_ellipsoid_cb = tk.Checkbutton(self.main_control_frame, text='Show ellipsoid',
                                                  variable=self.ellipsoid_selection_var, anchor='w', command=self.render_views)
        self.select_ellipsoid_cb.place(x=20, y=245, width=206 - 20, height=20)

        self.select_mesh_cb = tk.Checkbutton(self.main_control_frame, text='Show mesh',
                                                  variable=self.mesh_selection_var, anchor='w', command=self.render_views)
        self.select_mesh_cb.place(x=20, y=266, width=206 - 20, height=20)

        self.mesh_on_top_cb = tk.Checkbutton(self.main_control_frame, text='Mesh in top layer',
                                             variable=self.mesh_on_top_selection_var, anchor='w', command=self.render_views)
        self.mesh_on_top_cb.place(x=20, y=287, width=206 - 20, height=20)

        self.select_camera_cb = tk.Checkbutton(self.main_control_frame, text='Show camera',
                                             variable=self.camera_selection_var, anchor='w', command=self.render_main_view)
        self.select_camera_cb.place(x=20, y=308, width=206 - 20, height=20)

        self.select_atmosphere_cb = tk.Checkbutton(self.main_control_frame, text='Render atmospheric haze',
                                             variable=self.atm_selection_var, anchor='w', command=self.render_views)
        self.select_atmosphere_cb.place(x=238, y=370, width=206 - 20, height=20)
        self.select_at_horizon_atmosphere_cb = tk.Checkbutton(self.main_control_frame, text='Haze only at horizon',
                                             variable=self.atm_at_horizon_selection_var, anchor='w', command=self.render_views)
        self.select_at_horizon_atmosphere_cb.place(x=238, y=391, width=206 - 20, height=20)

        self.set_atm_color_controls()

        tk.Label(self.main_control_frame, text='  Render Resolution:', justify=tk.LEFT, anchor='w').place(x=218, y=281, width=206, height=20)
        tk.Label(self.main_control_frame, text='x', justify=tk.LEFT, anchor='w').place(x=310, y=301, width=10, height=20)

        
        self.render_res_x_entry = tk.Entry(self.main_control_frame, width=7, justify=tk.RIGHT, textvariable=self.render_res_x_var)
        self.render_res_y_entry = tk.Entry(self.main_control_frame, width=7, justify=tk.RIGHT, textvariable=self.render_res_y_var)
        self.render_res_x_entry.place(x=263, y=301)
        self.render_res_y_entry.place(x=321 , y=301)
    
        def update_render_res(*kaka):
            entry_vars = [self.render_res_x_var, self.render_res_y_var]
            for k, entry_var in enumerate(entry_vars):
                entry_string = entry_var.get()
                if entry_string == '':
                    continue
                if not entry_string.isdigit():
                    entry_string = '1'
                num_leading_zeros = len(entry_string) - len(str(int(entry_string)))
                val = int(entry_string)
                val = max(val, 1)
                entry_var.set('0' * num_leading_zeros + str(val))
            self.render_views()

        self.render_res_x_var.trace("w", update_render_res)
        self.render_res_y_var.trace("w", update_render_res)
        self.select_supersampling = tk.Checkbutton(self.main_control_frame, text='Enable 2x supersampling',
                                             variable=self.supersampling_var, anchor='w')
        self.select_supersampling.place(x=238, y=322, width=206 - 20, height=20)

        self.render_btn = tk.Button(self.main_control_frame, text='Render image', command=self.render_final_view)
        self.render_btn.place(x=218, y=342, width=206, height=23)

        self.set_view_cam_controls()
        self.reset_controls()


    def set_view_cam_latitude(self, latitude):
        self.cam_view_camera_pos, self.cam_view_camera_mat = fix_cam_pos_latitude(self.cam_view_camera_pos, self.cam_view_camera_mat, np.clip(latitude, -89.99999, 89.99999))
        if self.roll_lock_var.get() == 1:
            self.cam_view_camera_mat = fix_view_cam_roll(self.cam_view_camera_pos, self.cam_view_camera_mat, self.view_cam_roll_var.get())
        self.render_views()


    def set_view_cam_longitude(self, longitude):
        self.cam_view_camera_pos, self.cam_view_camera_mat = fix_cam_pos_longitude(self.cam_view_camera_pos, self.cam_view_camera_mat, longitude)
        self.render_views()


    def set_view_cam_height(self, height):
        self.cam_view_camera_pos = fix_cam_pos_height(self.cam_view_camera_pos, height)
        self.render_views()


    def set_view_cam_roll(self, event):
        val = self.view_cam_roll_var.get()
        self.cam_view_camera_mat = fix_view_cam_roll(self.cam_view_camera_pos, self.cam_view_camera_mat, val)
        self.render_views()

    def set_cam_to_juno_pos(self):
        if self.currently_active_loaded_image is not None:
            self.cam_view_camera_pos = self.renderer.get_juno_pos(self.currently_active_loaded_image).copy()
            self.update_cam_view_controls()
            self.render_views()

    def update_cam_view_controls(self):
        latitude, longitude, height, roll = get_camera_params_from_pos_mat(self.cam_view_camera_pos, self.cam_view_camera_mat)
        latitude, longitude, height = np.round(latitude, 4), np.round(longitude, 4), np.round(height, 1)
        self.view_cam_latitude_var.set(latitude)
        self.view_cam_latitude_var_entry.set(str(latitude))
        self.view_cam_longitude_var.set(longitude)
        self.view_cam_longitude_var_entry.set(str(longitude))
        self.view_cam_roll_var.set(roll)
        self.view_cam_height_var.set(height)
        self.view_cam_height_log_var.set(np.log(height + 1.0) - 2.3978952728)
        self.view_cam_height_var_entry.set(str(height))


    def show_tutorial(self):
        window = tk.Toplevel(self.root)
        window.geometry('950x800')
        window.geometry('+400+100')
        window.title('How to use')
        window.resizable(False, False)
        window.grab_set()

        main_label = tk.Label(window, text=tutorial_text, anchor='nw', justify=tk.LEFT)
        main_label.place(x=20, y=10, width=910, height=760)

        close_button = tk.Button(window, text='Okay', command=window.destroy)
        close_button.place(x=880, y=760, width=60, height=30)


    def set_view_cam_controls(self):
        def set_cam_latitude_slider(event):
            val = self.view_cam_latitude_var.get()
            val = np.round(val, 4)
            self.view_cam_latitude_var.set(val)
            self.view_cam_latitude_var_entry.set(str(val))
            self.set_view_cam_latitude(val)

        def set_cam_latitude_entry(*kaka):
            val = self.view_cam_latitude_var_entry.get()
            try:
                val = float(val)
            except ValueError:
                return
            val = np.clip(val, -90, 90)
            val = np.round(val, 4)
            self.view_cam_latitude_var.set(val)
            self.view_cam_latitude_var_entry.set(str(val))
            self.view_cam_latitude_entry.select_clear()

        self.cam_focal_length_label = tk.Label(self.main_control_frame, textvariable=self.cam_focal_length_text_var, anchor='w')
        self.main_view_focal_length_label = tk.Label(self.main_control_frame, textvariable=self.main_view_focal_length_text_var, anchor='w')
        self.cam_focal_length_label.place(x=218, y=450, width=206)
        self.main_view_focal_length_label.place(x=0, y=450, width=206)


        self.view_cam_latitude_slider = tk.Scale(self.main_control_frame, from_=-90, to=90, orient='horizontal', variable=self.view_cam_latitude_var,
                                            resolution=-1, label='Camera latitude:', width=13, length=200, showvalue=0, command=set_cam_latitude_slider)
        self.view_cam_latitude_slider.place(x=218, y=0 + 17, width=206)
        self.view_cam_latitude_var_entry.trace("w", set_cam_latitude_entry)
        self.view_cam_latitude_entry = tk.Entry(self.main_control_frame, width=13, justify=tk.RIGHT,
                                            textvariable=self.view_cam_latitude_var_entry)
        self.view_cam_latitude_entry.place(x=340, y=2 + 17, width=80)

        def set_cam_longitude_slider(event):
            val = self.view_cam_longitude_var.get()
            val = np.round(val, 4)
            self.view_cam_longitude_var.set(val)
            self.view_cam_longitude_var_entry.set(str(val))
            self.set_view_cam_longitude(val)

        def set_cam_longitude_entry(*kaka):
            val = self.view_cam_longitude_var_entry.get()
            try:
                val = float(val)
            except ValueError:
                return
            val = (val + 180) % 360 - 180
            val = np.round(val, 4)
            self.view_cam_longitude_var.set(val)
            self.view_cam_longitude_var_entry.set(str(val))
            self.view_cam_longitude_entry.select_clear()

        self.view_cam_longitude_slider = tk.Scale(self.main_control_frame, from_=-180, to=180, orient='horizontal', variable=self.view_cam_longitude_var,
                                            resolution=-1, label='Camera longitude:', width=13, length=200, showvalue=0, command=set_cam_longitude_slider)
        self.view_cam_longitude_slider.place(x=218, y=40 + 17, width=206)
        self.view_cam_longitude_var_entry.trace("w", set_cam_longitude_entry)
        self.view_cam_longitude_entry = tk.Entry(self.main_control_frame, width=13, justify=tk.RIGHT,
                                            textvariable=self.view_cam_longitude_var_entry)
        self.view_cam_longitude_entry.place(x=340, y=40 + 2 + 17, width=80)

        def set_cam_height_slider(event):
            val = self.view_cam_height_log_var.get() + 2.3978952728
            val = np.exp(val) - 1
            val = np.round(val, 4)
            self.view_cam_height_var.set(val)
            self.view_cam_height_log_var.set(np.log(val + 1) - 2.3978952728)
            self.view_cam_height_var_entry.set(str(val))
            self.set_view_cam_height(val)

        def set_cam_height_entry(*kaka):
            val = self.view_cam_height_var_entry.get()
            try:
                val = float(val)
            except ValueError:
                return
            val = max(val, 10)
            val = np.round(val, 1)
            self.view_cam_height_var.set(val)
            self.view_cam_height_log_var.set(np.log(val + 1) - 2.3978952728)
            self.view_cam_height_var_entry.set(str(val))
            self.view_cam_height_entry.select_clear()

        self.view_cam_height_slider = tk.Scale(self.main_control_frame, from_=0, to=13, orient='horizontal', variable=self.view_cam_height_log_var,
                                            resolution=-1, label='Camera height:', width=13, length=200, showvalue=0, command=set_cam_height_slider)
        self.view_cam_height_slider.place(x=218, y=80 + 17, width=206)
        self.view_cam_height_var_entry.trace("w", set_cam_height_entry)
        self.view_cam_height_entry = tk.Entry(self.main_control_frame, width=13, justify=tk.RIGHT,
                                            textvariable=self.view_cam_height_var_entry)
        self.view_cam_height_entry.place(x=340, y=80 + 2 + 17, width=80)

        self.view_cam_roll_slider = tk.Scale(self.main_control_frame, from_=-180, to=180, orient='horizontal', variable=self.view_cam_roll_var,
                                            resolution=-1, label='Camera roll angle:', width=13, length=200, command=self.set_view_cam_roll)
        self.view_cam_roll_slider.place(x=218, y=140, width=206)
        self.view_cam_roll_cb = tk.Checkbutton(self.main_control_frame, variable=self.roll_lock_var, text='Lock roll angle', anchor='w')
        self.view_cam_roll_cb.place(x=238, y=210, width=206 - 20, height=20)

        self.set_cam_to_juno_pos_btn = tk.Button(self.main_control_frame, text='Set camera to juno position', command=self.set_cam_to_juno_pos)
        self.set_cam_to_juno_pos_btn.place(x=218, y=245, width=206, height=23)


    #def add_to_brdf(self):
    #    self.renderer.brdf_maker.add_to_brdf(self.loaded_image_properties)


    #def show_current_brdf(self):
    #    brdf_cum, brdf_count = self.renderer.brdf_maker.get_brdf_cum_count()
    #    brdf = brdf_cum / np.maximum(brdf_count, 1)
    #    plt.xlim(0, brdf.shape[1])
    #    plt.ylim(0, brdf.shape[0])
    #    plt.imshow(brdf)
    #    plt.show()


    #def save_current_brdf(self):
    #    brdf_cum, brdf_count = self.renderer.brdf_maker.get_brdf_cum_count()
    #    np.save('BRDF_cum.npy', brdf_cum)
    #    np.save('brdf_count.npy', brdf_count)


    def pack_widgets(self):
        self.menu = tk.Menu(self.root, bg='gray')
        self.root.config(menu=self.menu)
        self.data_menu = tk.Menu(self.menu, tearoff=0)

        self.extras_menu = tk.Menu(self.menu, tearoff=0)
        self.extras_menu.add_command(label='Show Tutorial', command=self.show_tutorial)
        #self.extras_menu.add_separator()
        #self.extras_menu.add_command(label='Add to BDRF', command=self.add_to_brdf)
        #self.extras_menu.add_command(label='Show current BDRF', command=self.show_current_brdf)
        #self.extras_menu.add_command(label='Save current BDRF', command=self.save_current_brdf)

        self.data_menu.add_command(label='Load Perijove Data', command=self.select_perijove)
        self.data_menu.add_separator()
        self.data_menu.add_command(label='Export Main View', command=self.export_main_view)
        self.data_menu.add_command(label='Export Camera View', command=self.render_final_view)

        self.menu.add_cascade(label="Data", menu=self.data_menu)
        self.menu.add_cascade(label='Extras', menu=self.extras_menu)

        
        self.image_list_frame = tk.Frame(self.root, height=900, bg='darkgray')
        self.image_list_canvas = tk.Canvas(self.image_list_frame, width=256, height=900, bg='gray64', bd=-2)

        self.image_list_scrollbar = tk.Scrollbar(self.image_list_frame, orient='vertical', bg='gray64')
        self.image_list_scrollbar.config(command=self.image_list_canvas.yview)
        self.image_list_canvas.config(yscrollcommand=self.image_list_scrollbar.set)

        self.image_list_canvas.pack(side='left')
        self.image_list_scrollbar.pack(side='right', fill='y')

        self.image_list_frame.place(x=0, y=0, height=900, width=274)#.grid(row=0, rowspan=3, column=0)


        self.main_view_canvas = tk.Canvas(self.root, width=900, height=900, bg='black', bd=-2)
        self.init_main_view_camera()
        self.bind_main_view_controls()
        self.main_view_canvas.place(x=274, y=0, height=900, width=900)#.grid(row=0, rowspan=3, column=1)


        self.cam_view_canvas = tk.Canvas(self.root, width=424, height=424, bg='black', bd=-2)
        self.init_cam_view_camera()
        self.bind_cam_view_controls()
        self.cam_view_canvas.place(x=1175 , y=0, height=424, width=424)#.grid(row=0, column=3, sticky='en')

        self.main_control_frame = tk.Frame(self.root, width=424, height=475)
        self.main_control_frame.pack_propagate(False)
        self.pack_main_control_frame()
        self.main_control_frame.place(x=1175 , y=425, height=475, width=424)#.grid(row=1, column=3, sticky='news')

        



    def mainloop(self):
        self.root.mainloop()

