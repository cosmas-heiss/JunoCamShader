import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import moderngl
import time
import json
import os
import cv2

from SPICEHandler import SpiceDataHandler
from Util import build_ball, get_cam_wf_lines


 
CAMERA_SIZE = 0.02


class ViewRenderer:
    def __init__(self, gui_object=None, spice_data_handler=None):
        self.context = moderngl.create_context(standalone=True, require=430)
        self.context.enable(moderngl.DEPTH_TEST)

        self.group_size = 16

        self.resolutions = [(900, 900), (424, 424)]

        self.program_render_view = self.context.compute_shader(self.load_shader_file('ViewRenderShader.glsl'))
        self.program_wireframe = self.context.program(vertex_shader=self.load_shader_file('LineVertexShader.glsl'),
                                                      fragment_shader=self.load_shader_file('LineFragementShader.glsl'))
        self.program_composite = self.context.compute_shader(self.load_shader_file('CompositeShader.glsl'))

        self.BAD_PIXEL_MASK = (np.array(Image.open("./BAD_PIXEL_MASK.png")) == 0).astype(np.uint8)
        self.images_info_dict = {}
        self.global_info_dict = {'height_rel_offset_value': 1.0}
        self.big_raw_image = None

        self.atm_color = (138.0 / 255.0, 107.0 / 255.0, 45.0 / 255.0)

        self.spice_data_handler = spice_data_handler
        self.gui_object = gui_object

        self.set_up_buffers()
        self.set_up_wireframe_buffers()

        #self.brdf_maker = BRDFMaker(self.context, self)


    def set_up_buffers(self):
        brdf_data = np.load('BDRF.npy').astype(np.float32)#np.array(Image.open('./BRDF.png')).astype(np.float32) / 255.0
        self.brdf_texture = self.context.texture((512, 512), 1, dtype='f4', data=brdf_data[:, :].copy())

        self.black_image = [np.zeros((*res, 3), dtype=np.uint8) for res in self.resolutions]

        self.out_tex_image = [self.context.texture(res, 4, dtype='f4') for res in self.resolutions]
        self.out_tex_ellipsoid = [self.context.texture(res, 4, dtype='f4') for res in self.resolutions]
        self.fog_tex = [self.context.texture(res, 4, dtype='f4') for res in self.resolutions]
        self.frame_buffer = [self.context.simple_framebuffer(res) for res in self.resolutions]
        self.mesh_cam_tex = [self.context.texture(res, 3, dtype='u1') for res in self.resolutions]
        self.out_tex = [self.context.texture(res, 4, dtype='u1') for res in self.resolutions]


    def set_up_wireframe_buffers(self):
        self.vertex_buffer = self.context.buffer(data=build_ball(16, 16, 30).astype(np.float32))
        example_cam_lines = get_cam_wf_lines(np.zeros(3), np.eye(3), 1.0, 1.0, 1.0)

        ball_vertices = build_ball(16, 16, 30).astype(np.float32)
        ball_vertices = np.pad(ball_vertices, ((0, 0), (0, 1)))
        self.ball_buffer_length = len(ball_vertices) * 4 * 4
        self.vertex_buffer = self.context.buffer(data=np.zeros((len(ball_vertices) + len(example_cam_lines), 4), dtype=np.float32))
        self.vertex_buffer.write(ball_vertices)
        self.vertex_array = self.context.vertex_array(self.program_wireframe, self.vertex_buffer, 'vert_pos', 'cam_or_mesh')


    def load_shader_file(self, path):
        with open(path, 'r') as f:
            out = f.read()
        return out

    def set_gui_object(self, gui_object):
        self.gui_object = gui_object


    def set_height_offset(self, height_offset):
        relative_offset = (69911 + height_offset) / 69911
        self.global_info_dict['height_rel_offset_value'] = relative_offset


    def set_timing_buffer(self, image_settings_dict, for_brdf=False):
        buffer_data = np.zeros((self.get_total_num_framelets(), 4, 4), dtype=np.float32)
        image_index_data = np.zeros(len(self.images_info_dict), dtype=np.int32)

        for img_name in self.images_info_dict:
            image_info_dict = self.images_info_dict[img_name]
            stripe_selection = image_settings_dict[img_name]['framelet_list']

            start_time = image_info_dict['start_time'] + image_settings_dict[img_name]['time_offset']
            interframe_delay = image_info_dict['interframe_delay'] + image_settings_dict[img_name]['interframe_delay']
            num_framelets = image_info_dict['num_framelets']

            image_index_data[image_info_dict['image_index']] = num_framelets
     
            times = [start_time + i * interframe_delay for i in range(num_framelets)]

            sun_directions = self.spice_data_handler.get_sun_direction(times)
            framelet_inv_cam_matrices, framelet_cam_positions = self.spice_data_handler.get_inv_orients_positions(times)

            start_framelet = image_info_dict['framelet_range'][0]
            for k, (inv_cam_mat, cam_pos, sun_dir) in enumerate(zip(framelet_inv_cam_matrices, framelet_cam_positions, sun_directions)):
                buffer_data[start_framelet + k, 0, :3] = cam_pos
                buffer_data[start_framelet + k, 1:4, :3] = inv_cam_mat
                buffer_data[start_framelet + k, 1:4, 3] = sun_dir
                if for_brdf:
                    buffer_data[start_framelet + k, 0, 3] = image_info_dict['exposure_time']
                else:
                    if stripe_selection[k]:
                        buffer_data[start_framelet + k, 0, 3] = 1.0
                    else:
                        buffer_data[start_framelet + k, 0, 3] = 0.0

        self.cam_inv_mat_pos_buffer.write(buffer_data)
        self.image_index_buffer.write(image_index_data)
        
        


    def load_raw_image_into_texture(self, perijove_path, image_name):
        with open(os.path.join(perijove_path, 'images_info', image_name + '.json'), 'r') as file_content:
            meta_info_dict = json.load(file_content)
        image_info_dict = {}
        image_info_dict['start_time'] = self.spice_data_handler.convert_time_str2et(meta_info_dict['START_TIME'])
        image_info_dict['interframe_delay'] = float(meta_info_dict["INTERFRAME_DELAY"].split()[0])
        image_info_dict['num_framelets'] = meta_info_dict["LINES"] // 384
        mid_time = image_info_dict['start_time'] + image_info_dict['num_framelets'] * 0.5 * image_info_dict['interframe_delay']
        image_info_dict['juno_pos_approx'] = np.array(self.spice_data_handler.get_orients_positions([mid_time])[1][0])
        image_info_dict['exposure_time'] = float(meta_info_dict['EXPOSURE_DURATION'].split(' ')[0])


        img_ar = np.array(Image.open(os.path.join(perijove_path, 'images', image_name + '.png')))
        s1, s2 = img_ar.shape
        new_img_ar = np.zeros((s1 // 16, s2 * 16))
        for k in range(image_info_dict['num_framelets']):
            self.gui_object.update_progress_bar(99.999 / image_info_dict['num_framelets'], text='Filling Dead Pixels..')
            new_img_ar[k * 24: (k + 1) * 24] = cv2.inpaint(img_ar[k * 384: (k + 1) * 384], self.BAD_PIXEL_MASK, 3, cv2.INPAINT_NS).reshape(24, 16 * 1648)

        new_img_ar = (new_img_ar * np.sqrt(5.2 / image_info_dict['exposure_time']) / 255).astype(np.float32)
        if self.big_raw_image is None:
            self.big_raw_image = new_img_ar
            image_info_dict['big_raw_image_lines'] = (0, new_img_ar.shape[0])
            image_info_dict['framelet_range'] = (0, image_info_dict['num_framelets'])
            image_info_dict['image_index'] = 0
        else:
            prev_length = self.big_raw_image.shape[0]
            self.big_raw_image = np.concatenate((self.big_raw_image, new_img_ar), axis=0)
            image_info_dict['big_raw_image_lines'] = (prev_length, self.big_raw_image.shape[0])
            image_info_dict['framelet_range'] = (self.get_total_num_framelets(), self.get_total_num_framelets() + image_info_dict['num_framelets'])
            image_info_dict['image_index'] = len(self.images_info_dict)

        if hasattr(self, 'raw_image_texture'):
            self.raw_image_texture.release()
        
        self.raw_image_texture = self.context.texture(self.big_raw_image.shape[::-1], 1, dtype='f4', data=self.big_raw_image)

        if hasattr(self, 'cam_inv_mat_pos_buffer'):
            self.cam_inv_mat_pos_buffer.release()
        self.cam_inv_mat_pos_buffer = self.context.buffer(reserve=(self.get_total_num_framelets() + image_info_dict['num_framelets']) * 4 * 4 * 4, dynamic=False)

        if hasattr(self, 'image_index_buffer'):
            self.image_index_buffer.release()
        self.image_index_buffer = self.context.buffer(reserve=(len(self.images_info_dict) + 1) * 4, dynamic=False)

        return image_info_dict


    def get_total_num_framelets(self):
        return sum([x['num_framelets'] for x in self.images_info_dict.values()])


    def load_image(self, perijove_path, image_name):
        image_info_dict = self.load_raw_image_into_texture(perijove_path, image_name)
        image_info_dict['perijove_path'] = perijove_path
        image_info_dict['image_name'] = image_name
        
        self.program_render_view['num_images'] = len(self.images_info_dict) + 1
        self.program_render_view['sun_direction'] = tuple(x for x in self.spice_data_handler.get_sun_direction(image_info_dict['start_time']))
        self.images_info_dict[image_name] = image_info_dict



    def clear_loaded_images(self):
        self.images_info_dict = {}
        self.big_raw_image = None

        if hasattr(self, 'raw_image_texture'):
            self.raw_image_texture.release()
            del self.raw_image_texture

        if hasattr(self, 'cam_inv_mat_pos_buffer'):
            self.cam_inv_mat_pos_buffer.release()
            del self.cam_inv_mat_pos_buffer

        if hasattr(self, 'image_index_buffer'):
            self.image_index_buffer.release()
            del self.image_index_buffer


    def unload_image(self, image_name):
        image_info_dict = self.images_info_dict[image_name]
        self.images_info_dict.pop(image_name)

        if len(self.images_info_dict) == 0:
            self.big_raw_image = None
            return

        self.raw_image_texture.release()
        self.cam_inv_mat_pos_buffer.release()
        self.image_index_buffer.release()

        self.program_render_view['num_images'] = len(self.images_info_dict)

        start_end = image_info_dict['big_raw_image_lines']
        self.big_raw_image = np.concatenate((self.big_raw_image[:start_end[0]], self.big_raw_image[start_end[1]:]), axis=0)
        for other_image_info in self.images_info_dict.values():
            other_start_end = other_image_info['big_raw_image_lines']
            other_framelet_range = other_image_info['framelet_range']
            if other_start_end[0] > start_end[0]:
                tmp_length = start_end[1] - start_end[0]
                other_image_info['big_raw_image_lines'] = (other_start_end[0] - tmp_length, other_start_end[1] - tmp_length)
                other_image_info['framelet_range'] = (other_framelet_range[0] - image_info_dict['num_framelets'],
                                                      other_framelet_range[1] - image_info_dict['num_framelets'])
                other_image_info['image_index'] -= 1
        
        self.raw_image_texture = self.context.texture(self.big_raw_image.shape[::-1], 1, dtype='f4', data=self.big_raw_image)
        self.cam_inv_mat_pos_buffer = self.context.buffer(reserve=self.get_total_num_framelets() * 4 * 4 * 4, dynamic=False)
        self.image_index_buffer = self.context.buffer(reserve=len(self.images_info_dict) * 4, dynamic=False)


    def get_num_framelets(self, img_name):
        return self.images_info_dict[img_name]['num_framelets']


    def get_juno_pos(self, img_name):
        return self.images_info_dict[img_name]['juno_pos_approx']


    def set_atm_color(self, atm_color):
        self.atm_color = tuple(x / 255.0 for x in atm_color)


    def render_image_and_ellipsoid(self, camera_pos, camera_mat, view_index, image_settings_dict, wf_render_mode, blending_params):
        res = self.resolutions[view_index]
        self.program_render_view['JUPITER_EQUATORIAL_RADIUS'] = 71492.0 * self.global_info_dict['height_rel_offset_value']
        self.program_render_view['JUPITER_POLAR_RADIUS'] = 66854.0 * self.global_info_dict['height_rel_offset_value']
        self.program_render_view['camera_mat'] = tuple(x for y in camera_mat for x in y)
        self.program_render_view['camera_pos'] = tuple(x for x in camera_pos)
        self.program_render_view['resolution'] = res
        self.program_render_view['fog_base_color'] = self.atm_color
        self.program_render_view['atm_reduced'] = wf_render_mode[5] == 1
        self.program_render_view['mixing_strength'] = blending_params

        self.set_timing_buffer(image_settings_dict)

        self.raw_image_texture.use(0)
        self.image_index_buffer.bind_to_storage_buffer(1)
        self.cam_inv_mat_pos_buffer.bind_to_storage_buffer(2)
        self.out_tex_image[view_index].bind_to_image(3)
        self.out_tex_ellipsoid[view_index].bind_to_image(4)
        self.fog_tex[view_index].bind_to_image(5)
        self.brdf_texture.use(6)
        self.program_render_view.run(group_x=int(np.ceil(res[0] / self.group_size)), group_y=int(np.ceil(res[1] / self.group_size)))
        self.context.finish()


    def set_vertex_buffer(self, camera_pos, extra_info, aspect_ratio):
        self.vertex_buffer.orphan()
        extra_lines = np.zeros((2 * 12, 4), dtype=np.float32)
        if extra_info is not None:
            cam_size = np.linalg.norm(camera_pos - extra_info[0]) * CAMERA_SIZE / extra_info[2]
            extra_lines = get_cam_wf_lines(extra_info[0], extra_info[1], extra_info[3], cam_size, aspect_ratio,
                radius_factor=self.global_info_dict['height_rel_offset_value'])
            extra_lines = np.pad(extra_lines, ((0, 0), (0, 1)), constant_values=1.0)

        self.vertex_buffer.write(extra_lines, offset=self.ball_buffer_length)


    def render_mesh_and_cam(self, camera_pos, camera_mat, view_index, cam_view_info, wf_render_mode, aspect_ratio, resolution=None):
        res = self.resolutions[view_index]
        if resolution is not None:
            res = resolution
        self.program_wireframe['JUPITER_EQUATORIAL_RADIUS'] = 71492.0 * self.global_info_dict['height_rel_offset_value']
        self.program_wireframe['JUPITER_POLAR_RADIUS'] = 66854.0 * self.global_info_dict['height_rel_offset_value']
        self.program_wireframe['inv_camera_mat'] = tuple(x for y in np.linalg.inv(camera_mat) for x in y)
        self.program_wireframe['cam_pos'] = tuple(x for x in camera_pos)
        self.program_wireframe['show_cam'] = wf_render_mode[3] == 1
        self.program_wireframe['show_ellipsoid'] = wf_render_mode[0] == 1
        self.program_wireframe['aspect_ratio'] = res[0] / res[1]

        self.set_vertex_buffer(camera_pos, cam_view_info, aspect_ratio)
        
        self.frame_buffer[view_index].use()
        self.frame_buffer[view_index].clear(0.0, 0.0, 0.0, 0.0, depth=1.0)

        self.vertex_array.render(moderngl.LINES)
        out_data = np.frombuffer(self.frame_buffer[view_index].read(), 'uint8').reshape(res[1], res[0], 3)
        self.mesh_cam_tex[view_index].write(out_data)


    def composite_image(self, view_index, wf_render_mode, aspect_ratio, camera_pos, camera_mat, super_sampling=False):
        res = self.resolutions[view_index]
        self.program_composite['wf_render_mode'] = wf_render_mode[:4]
        self.program_composite['render_atm'] = wf_render_mode[4] == 1
        self.program_composite['resolution'] = res
        self.program_composite['super_sampling'] = super_sampling
        self.program_composite['aspect_ratio'] = aspect_ratio if view_index == 1 else 1.0

        self.out_tex_image[view_index].use(0)
        self.out_tex_ellipsoid[view_index].use(1)
        self.mesh_cam_tex[view_index].use(2)
        self.fog_tex[view_index].use(3)
        self.out_tex[view_index].bind_to_image(4)
        self.program_composite.run(group_x=int(np.ceil(res[0] / self.group_size)), group_y=int(np.ceil(res[1] / self.group_size)))
        self.context.finish()


    def render_view(self, camera_pos, camera_mat, view_index, image_settings_dict, blending_params, render_res=(1, 1), wf_render_mode=(1, 0, 0, 1, 1, 1), cam_view_info=None):
        if len(self.images_info_dict) == 0:
            return self.black_image[view_index]

        self.render_image_and_ellipsoid(camera_pos, camera_mat, view_index, image_settings_dict, wf_render_mode, blending_params)
        self.render_mesh_and_cam(camera_pos, camera_mat, view_index, cam_view_info, wf_render_mode, render_res[0] / render_res[1])
        self.composite_image(view_index, wf_render_mode, render_res[0] / render_res[1], camera_pos, camera_mat)

        res = self.resolutions[view_index]
        output_array = np.frombuffer(self.out_tex[view_index].read(), 'uint8').copy().reshape(res[1], res[0], 4)[:, :, :3]

        return output_array


    def render_final_view(self, camera_pos, camera_mat, image_settings_dict, blending_params, res, wf_render_mode=(1, 0, 0, 1, 1, 1), super_sampling=True):
        old_res = res
        if super_sampling:
            res = (2 * res[0], 2 * res[1])

        if len(self.resolutions) == 3:
            self.resolutions.pop()
        self.resolutions.append(res)
        texture_lists = [self.out_tex_image, self.out_tex_ellipsoid, self.fog_tex, self.frame_buffer, self.mesh_cam_tex, self.out_tex]
        new_textures = [self.context.texture(res, 4, dtype='f4') for i in range(3)]
        new_textures += [self.context.simple_framebuffer(old_res), self.context.texture(old_res, 3, dtype='u1'), self.context.texture(old_res, 4, dtype='u1')]
        for texture_list, new_texture in zip(texture_lists, new_textures):
            if len(texture_list) == 3:
                texture_list.pop()
            texture_list.append(new_texture)


        self.render_image_and_ellipsoid(camera_pos, camera_mat, 2, image_settings_dict, wf_render_mode, blending_params)
        self.render_mesh_and_cam(camera_pos, camera_mat, 2, None, wf_render_mode, 1.0, resolution=old_res)
        self.composite_image(2, wf_render_mode, 1.0, camera_pos, camera_mat, super_sampling=super_sampling) # todo also give cam pos and mat

        output_array = np.frombuffer(self.out_tex[2].read(), 'uint8').copy().reshape(old_res[1], old_res[0], 4)[:, :, :3]
        
        for texture_list in texture_lists:
            texture_list[-1].release()

        return output_array