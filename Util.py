import numpy as np
import time

JUPITER_EQUATORIAL_RADIUS = 71492  # km
JUPITER_POLAR_RADIUS = 66854  # km




tutorial_text = """1) Loading data:
        Select "Load Perijove Data" from the "Data" menu. You will be presented with a list of available perijove folders. If no folders are present or you want to download
        image data from an additional mission phase, click "Download New".

        This will open a prompt to enter the perijove number of the image set to download. Upon downloading, a new folder will be created within the "./Data" subdirectory,
        where all raw images, image metadata files and SPICE kernels will be saved. During the download, an additional "./Data/tmp" subdirectory is created for extraction.
        It will be deleted after the downloading process.

        Note that when downloading data from recent flybys of the Juno spaceprobe, the corresponding spacecraft data kernels might not be publicly available yet. This
        results in errors from the SPICE toolkit.

        Select a perijove by clicking on it in the list and clicking the "Select" button.

2) Loading a raw image:
        After sucessfully loading the raw images from a perijove, they will be presented in a list on the left. Double-click one of these elements to load the image.
        You can also load multiple images and deselect some. The parameters are always for the image in the red box.

3) Main view and camera view:
        When loading a raw image, its projection of it will be shown in two view windows: The "Main view" on the left and the "Camera view" in the top right. The position
        and orientation of the camera used to generate the "Camera view" is indicated as a red wireframe model in the "Main view".

4) View Controls:
        The views of both the "Main view" window and the "Camera view" window can be rotated by clicking and dragging with the mouse. Scrolling will shift the camera
        forward/backward. Holding SHIFT and scrolling changes the focal length.

5) Projection Controls:
        The sliders "Imaging time offset", "Interframe delay offset" and "Surface height offset" can be used to adjust the projection of the raw image onto Jupiters surface.

        The "Imaging time offset" adjusts the time in Junos rotation when the individual slices of the image are taken. This is needed as the exact time of image acquisition
        is not always known due to random delays. If A black edge can be seen on one side of the projected image, this slider can be used to center the projection properly.

        The "Interframe delay offset" is used to adjust the time delay between the aquisition of individual stripes. If the image features chromatic aberration use this slider to
        correct it.

        For different reasons, these two adjustments might not be enough to properly project the image onto the surface of Jupiter. In this case, use the "Surface height offset"
        to adjust the height of the surface that the raw image stripes are projected on.

        The button "Select framelet subset" can be used to select only a subset of the raw image stripes to be projected onto Jupiters surface.

6) Camera Controls:
        The latitude, longitude and height above the surface can be adjusted using the sliders on the right. If needed, these values can also be manually inputted.
        The "Roll angle" slider adjusts the angle with respect to the horizon. Use "Lock roll angle" to fix this angle while moving the camera. The "Set camera to Juno position"
        sets the viewing camera to the position of the Juno spaceprobe at the time of image acquisition.

7) Rendering Controls:
        The solid ellipsoid to project onto, as well as the simulated atmospheric haze and a mesh signifying latitudes and longitudes can be turned on and off using the
        checkbuttons in the lower right. This applies to the "Main view" as well as the "Camera view".

        A render resolution can be inputted on the lower right. When choosing an aspect ration different to 1:1, the view slice will be indicated in the "Camera view" window
        and the red rectangle indicating the cameras field of view in the "Main view" will be adjusted accordingly. When clicking "Render image" or when selecting
        "Export Camera View" in the menu, the current camera view is rendered in the prescribed resolution and the image can be saved as a PNG file.

"""


def scroll_view_to_surface(cur_cam_pos, delta):
    pos = cur_cam_pos.copy()
    pos[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
    pos_norm = np.linalg.norm(pos)
    pos_normalized = pos / pos_norm

    scaled_dist_to_surface = pos_norm - JUPITER_EQUATORIAL_RADIUS

    scaled_dist_to_surface *= np.exp(-delta)
    scaled_dist_to_surface = max(scaled_dist_to_surface, 0)

    new_pos = pos_normalized * (scaled_dist_to_surface + JUPITER_EQUATORIAL_RADIUS)
    new_pos[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
    return new_pos



def get_normal_vector(pos):
    a, b = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS, JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
    normal_vector = pos.copy() * np.array([a, a, b])
    normal_vector /= np.linalg.norm(normal_vector)
    return normal_vector


def fix_view_cam_roll(cam_pos, cam_mat, fixed_roll):
    normal_vector = get_normal_vector(cam_pos)
    side_vector = np.cross(cam_mat[2], normal_vector)
    side_vector_norm = np.linalg.norm(side_vector)
    if side_vector_norm == 0.0:
        return cam_mat
    side_vector /= side_vector_norm
    up_vector = np.cross(side_vector, cam_mat[2])
    fixed_roll = fixed_roll * np.pi / 180.0
    cam_mat[1] = np.cos(fixed_roll) * up_vector - np.sin(fixed_roll) * side_vector
    cam_mat[0] = np.sin(fixed_roll) * up_vector + np.cos(fixed_roll) * side_vector
    return cam_mat


def get_intersections(pos, ray_dirs, radius_factor):
    a = JUPITER_EQUATORIAL_RADIUS * JUPITER_EQUATORIAL_RADIUS * radius_factor * radius_factor
    b = JUPITER_POLAR_RADIUS * JUPITER_POLAR_RADIUS * radius_factor * radius_factor
    bba = np.array((b, b, a))

    q1 = np.dot(bba, ray_dirs * ray_dirs)
    q2 = 2.0 * np.dot(bba * pos, ray_dirs)
    q3 = np.dot(pos * pos, bba) - a * b

    p = q2 / q1
    q = q3 / q1

    tmp = 0.25 * p * p - q
    s = np.where(tmp >= 0, -0.5 * p - np.sqrt(np.maximum(tmp, 0.0)), -1)
    return np.where(s[None, :] > 0, pos[:, None] + s[None, :] * ray_dirs, 0), s <= 0


def get_cam_wf_lines(cam_pos, cam_mat, cam_focal_length, cam_box_size, aspect_ratio, radius_factor=1.0):
    connect_dict = {(0., 0., 0.): [(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                    (1., 0., 0.): [(1., 1., 0.), (1., 0., 1.)],
                    (0., 1., 0.): [(1., 1., 0.), (0., 1., 1.)],
                    (0., 0., 1.): [(1., 0., 1.), (0., 1., 1.)],
                    (1., 1., 0.): [(1., 1., 1.)],
                    (1., 0., 1.): [(1., 1., 1.)],
                    (0., 1., 1.): [(1., 1., 1.)],
                    (0.5, 1.35, 0.25): [(0.5, 1.0, 0.25), (0.5, 1.35, 0.75)],
                    (0.5, 1.35, 0.75): [(0.5, 1.0, 0.75)],
                    (0.0, 0.5, 1.5): [(0.25, 0.5, 1.0), (0.15, 0.85, 1.5), (0.15, 0.15, 1.5)],
                    (1.0, 0.5, 1.5): [(0.75, 0.5, 1.0), (0.85, 0.85, 1.5), (0.85, 0.15, 1.5)],
                    (0.5, 0.0, 1.5): [(0.5, 0.25, 1.0), (0.15, 0.15, 1.5), (0.85, 0.15, 1.5)],
                    (0.5, 1.0, 1.5): [(0.5, 0.75, 1.0), (0.15, 0.85, 1.5), (0.85, 0.85, 1.5)]}

    lines = []
    for i in connect_dict:
        a = cam_pos + cam_mat.T.dot(np.array(i) - 0.5) * cam_box_size
        for j in connect_dict[i]:
            b = cam_pos + cam_mat.T.dot(np.array(j) - 0.5) * cam_box_size
            lines.append(a)
            lines.append(b)


    subdiv = 64
    cam_mat = cam_mat.copy()
    cam_mat[2] *= cam_focal_length
    ex_a, ex_b = np.zeros(3), np.zeros(3)

    x_extent, y_extent = min(aspect_ratio, 1.0) * 0.5, min(1.0 / aspect_ratio, 1.0) * 0.5
    corners = np.array([(-x_extent, -y_extent, 1.0), (x_extent, -y_extent, 1.0),
                        (x_extent, y_extent, 1.0), (-x_extent, y_extent, 1.0)])
    extra_lines = []
    for i in range(4):
        corner_a, corner_b = corners[i], corners[(i + 1) % 4]
        points = np.linspace(corner_a, corner_b, subdiv + 1, endpoint=True).T
        points, mask = get_intersections(cam_pos, cam_mat.T.dot(points), radius_factor * 1.001)
        for k in range(subdiv):
            point_a, point_b = points[:, k], points[:, k + 1]
            if mask[k] or mask[k + 1]:
                extra_lines.append(ex_a)
                extra_lines.append(ex_b)
            else:
                extra_lines.append(point_a)
                extra_lines.append(point_b)

    return np.array(lines + extra_lines).astype(np.float32)


def look_inward_mat(pos):
    pos_norm = np.linalg.norm(pos)
    if pos_norm == 0.0:
        return np.eye(3)
    z_dir = -pos / pos_norm
    x_dir = np.cross(z_dir, np.array([0.0, 0.0, 1.0]))
    x_dir_norm = np.linalg.norm(x_dir)
    if x_dir_norm == 0.0:
        x_dir = np.array([1.0, 0.0, 0.0])
    else:
        x_dir /= x_dir_norm
    y_dir = np.cross(z_dir, x_dir)
    return np.array([x_dir, -y_dir, z_dir])


def scroll_view_cam_view(cur_cam_pos, cam_mat, delta, fixed_roll):
    pos = cur_cam_pos.copy()
    pos[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
    pos_norm = np.linalg.norm(pos)
    scaled_dist_to_surface = pos_norm - JUPITER_EQUATORIAL_RADIUS

    step_size = scaled_dist_to_surface * 0.1 * delta

    cur_cam_pos += cam_mat[2] * step_size

    cur_cam_pos[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
    cur_cam_norm = np.linalg.norm(cur_cam_pos)
    if cur_cam_norm < JUPITER_EQUATORIAL_RADIUS:
        cur_cam_pos *= JUPITER_EQUATORIAL_RADIUS / cur_cam_norm
    cur_cam_pos[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS 

    if fixed_roll is not None:
        cam_mat = fix_view_cam_roll(cur_cam_pos, cam_mat, fixed_roll)

    return cur_cam_pos, cam_mat


def move_main_view(cur_cam_pos, delta):
    pos = cur_cam_pos
    r = np.linalg.norm(pos)

    surf_dist = np.linalg.norm(pos * np.array([1, 1, JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS])) - JUPITER_EQUATORIAL_RADIUS

    delta *= surf_dist * 0.000005
    alpha = np.arctan2(pos[0], pos[1])
    phi = np.arctan2(pos[2], np.linalg.norm(pos[:2]))
    alpha += delta[0]
    phi = np.clip(phi + delta[1], -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

    pos_new = np.array((np.sin(alpha) * np.cos(phi), np.cos(alpha) * np.cos(phi), np.sin(phi))) * r

    pos_new_alt = pos_new.copy()
    pos_new_alt[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
    safety_factor = JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos_new_alt)
    if safety_factor > 1.0:
        pos_new *= safety_factor

    return pos_new


def fix_cam_pos_latitude(cam_pos, cam_mat, latitude):
    pos_stretched = cam_pos.copy()
    pos_stretched[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS

    surf_point = pos_stretched * JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos_stretched)
    surf_point[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS

    cur_height = np.linalg.norm(cam_pos - surf_point)


    old_latitude = np.arctan2(cam_pos[2], np.linalg.norm(cam_pos[:2]))
    alpha = latitude * np.pi / 180.0 - old_latitude
    axis = np.array([cam_pos[1], -cam_pos[0], 0.])
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return cam_pos, cam_mat
    axis /= axis_norm
    rot_mat = np.cos(alpha) * np.eye(3) + np.sin(alpha) * np.cross(axis, -np.eye(3)) + (1 - np.cos(alpha)) * np.outer(axis, axis)

    cam_mat = cam_mat.dot(rot_mat.T)

    new_pos = rot_mat.dot(cam_pos)
    pos_stretched = new_pos.copy()
    pos_stretched[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS

    surf_point = pos_stretched * JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos_stretched)
    surf_point[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS
    new_pos = new_pos * (np.linalg.norm(surf_point) + cur_height) / np.linalg.norm(new_pos)
    return new_pos, cam_mat


def fix_cam_pos_longitude(cam_pos, cam_mat, longitude):
    old_longitude = np.arctan2(cam_pos[0], cam_pos[1]) * 180.0 / np.pi
    alpha = (longitude - old_longitude) * np.pi / 180.0
    a, b = np.cos(alpha), np.sin(alpha)
    rot_mat = np.array([[a, b, 0.], [-b, a, 0.], [0., 0., 1.]])
    cam_mat = cam_mat.dot(rot_mat.T)
    return rot_mat.dot(cam_pos), cam_mat


def fix_cam_pos_height(cam_pos, height):
    pos_stretched = cam_pos.copy()
    pos_stretched[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS

    surf_point = pos_stretched * JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(pos_stretched)
    surf_point[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS

    point_radius = np.linalg.norm(surf_point)

    cam_pos = cam_pos * (point_radius + height) / np.linalg.norm(cam_pos)

    return cam_pos


def move_cam_view_fixed_roll(cur_cam_pos, cur_cam_mat, delta, fixed_roll):
    normal_vector = get_normal_vector(cur_cam_pos)
    fixed_roll = fixed_roll * np.pi / 180.0

    rot_axis = normal_vector
    rot_amount = -delta[0] * np.cos(fixed_roll) - delta[1] * np.sin(fixed_roll)
    rot_mat = np.cos(rot_amount) * np.eye(3) + np.sin(rot_amount) * np.cross(rot_axis, -np.eye(3)) + (1 - np.cos(rot_amount)) * np.outer(rot_axis, rot_axis)
    cur_cam_mat = cur_cam_mat.dot(rot_mat)

    rot_axis = np.cross(normal_vector, cur_cam_mat[2])
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm == 0.0:
        return cur_cam_mat
    rot_axis /= rot_axis_norm
    rot_amount = delta[1] * np.cos(fixed_roll) - delta[0] * np.sin(fixed_roll)
    rot_mat = np.cos(rot_amount) * np.eye(3) + np.sin(rot_amount) * np.cross(rot_axis, -np.eye(3)) + (1 - np.cos(rot_amount)) * np.outer(rot_axis, rot_axis)
    cur_cam_mat = cur_cam_mat.dot(rot_mat)

    return cur_cam_mat


def move_cam_view(cur_cam_pos, cur_cam_mat, delta, fixed_roll):
    if fixed_roll is not None:
        return move_cam_view_fixed_roll(cur_cam_pos, cur_cam_mat, delta, fixed_roll)

    rot_axis = -cur_cam_mat[1] * delta[0] - cur_cam_mat[0] * delta[1]

    rot_amount = np.linalg.norm(delta)
    rot_axis /= np.linalg.norm(rot_axis)

    rot_mat = np.cos(rot_amount) * np.eye(3) + np.sin(rot_amount) * np.cross(rot_axis, -np.eye(3)) + (1 - np.cos(rot_amount)) * np.outer(rot_axis, rot_axis)
    return cur_cam_mat.dot(rot_mat)


def get_camera_params_from_pos_mat(cam_pos, cam_mat):
    latitude = np.arctan2(cam_pos[2], np.linalg.norm(cam_pos[:2])) * 180 / np.pi
    longitude = np.arctan2(cam_pos[0], cam_pos[1]) * 180 / np.pi

    normal_vector = get_normal_vector(cam_pos)
    axis = np.cross(normal_vector, cam_mat[2])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-4:
        roll = 0.0
    else:
        axis /= axis_norm
        up_vector = np.cross(axis, cam_mat[2])
        a, b = np.dot(cam_mat[0], axis), np.dot(cam_mat[0], up_vector)
        roll = -np.arctan2(b, -a) * 180 / np.pi

    stretched_pos = cam_pos.copy()
    stretched_pos[2] *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS
    surf_point = stretched_pos * JUPITER_EQUATORIAL_RADIUS / np.linalg.norm(stretched_pos)
    surf_point[2] *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS

    height = np.linalg.norm(cam_pos - surf_point)

    return latitude, longitude, height, roll




def build_ball(num_latitudes, num_longitudes, subdiv=30):
    # latitude is breitengrad, longitude ist laengengrad
    eps_a = 1e-3

    line_segments = []
    for alpha in np.linspace(-np.pi / 2, np.pi / 2, num_latitudes + 1, endpoint=False)[1:]:
        r, z = np.cos(alpha), np.sin(alpha)

        psis = np.linspace(0, 2 * np.pi, 2 * subdiv, endpoint=False)
        for i in range(2 * subdiv):
            psi1, psi2 = psis[i], psis[(i+1)%(2 * subdiv)]
            a = np.array([np.sin(psi1 - eps_a / subdiv) * r, np.cos(psi1 - eps_a / subdiv) * r, z])
            b = np.array([np.sin(psi2 + eps_a / subdiv) * r, np.cos(psi2 + eps_a / subdiv) * r, z])

            line_segments.append(a)
            line_segments.append(b)


    for psi in np.linspace(0, 2 * np.pi, num_longitudes, endpoint=False):
        alphas = np.linspace(-np.pi / 2, np.pi / 2, subdiv + 1)
        for i in range(subdiv):
            r1, z1 = np.cos(alphas[i] - eps_a / subdiv), np.sin(alphas[i] - eps_a / subdiv)
            r2, z2 = np.cos(alphas[i+1] + eps_a / subdiv), np.sin(alphas[i+1] + eps_a / subdiv)
            a = np.array([np.sin(psi) * r1, np.cos(psi) * r1, z1])
            b = np.array([np.sin(psi) * r2, np.cos(psi) * r2, z2])

            line_segments.append(a)
            line_segments.append(b)


    return np.array(line_segments)

