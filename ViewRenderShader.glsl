#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D input_raw_image;
layout(std430, binding = 1) coherent buffer image_index_buffer_kaka {int table[];} image_index_buffer;
layout(std430, binding = 2) coherent buffer junocam_view_buffer_kaka {mat4x4 table[];} junocam_view_buffer;
layout(binding = 3) uniform writeonly image2D output_image_tex;
layout(binding = 4) uniform writeonly image2D output_ellipsoid_tex;
layout(binding = 5) uniform writeonly image2D output_fog_tex;
layout(binding = 6) uniform sampler2D bdrf_tex;


uniform int num_images;
uniform mat3x3 camera_mat;
uniform vec3 camera_pos;
uniform ivec2 resolution;
uniform vec3 sun_direction;
uniform vec3 fog_base_color;
uniform bool atm_reduced;
uniform vec2 mixing_strength;

uniform float JUPITER_EQUATORIAL_RADIUS;
uniform float JUPITER_POLAR_RADIUS;


const float atm_lambda = 45.0; // km
const float cutoff_factor = 5.0;
const int num_steps = 32;
const float atm_base_density = 0.002; // per km

//# camera distortion parameters taken from
//# https://naif.jpl.nasa.gov/pub/naif/pds/data/jno-j_e_ss-spice-6-v1.0/jnosp_1000/data/ik/juno_junocam_v03.ti
//# INS-6150#_FOCAL_LENGTH/INS-6150#_PIXEL_SIZE 10.95637/0.0074, 10.95637/0.0074, 10.95637/0.0074
const float focal_length = 1480.59054054;
//# INS-6150#_DISTORTION_K1
const float k1 = -5.9624209455667325e-08;
//# INS-6150#_DISTORTION_K2
const float k2 = 2.7381910042256151e-14;

//# INS-6150#_DISTORTION_X and # INS-6150#_DISTORTION_Y
//# in lists for each color: 0: blue, 1: green, 2: red
const mat3x2 color_centers = mat3x2(vec2(814.21, 158.48), vec2(814.21, 3.48), vec2(814.21, -151.52));


vec4 get_intersection_point(vec3 pos, vec3 ray_dir) {
    float a = JUPITER_EQUATORIAL_RADIUS * JUPITER_EQUATORIAL_RADIUS;
    float b = JUPITER_POLAR_RADIUS * JUPITER_POLAR_RADIUS;
    vec3 bba = vec3(b, b, a);

    float q1 = dot(ray_dir * ray_dir, bba);
    float q2 = 2.0 * dot(pos * ray_dir, bba);
    float q3 = dot(pos * pos, bba) - a * b;

    float p = q2 / q1;
    float q = q3 / q1;

    float tmp = 0.25 * p * p - q;
    if (tmp >= 0.0) {
        float s = -0.5 * p - sqrt(tmp);
        return vec4(pos + s * ray_dir, step(0.0, s));
    } else {
        return vec4(0.0);
    }
}


vec3 get_ball_intersection_s(vec3 pos, vec3 ray_dir, float height) {
    float a = JUPITER_EQUATORIAL_RADIUS + height;
    a = a * a;
    float b = JUPITER_POLAR_RADIUS + height;
    b = b * b;
    vec3 bba = vec3(b, b, a);

    float q1 = dot(ray_dir * ray_dir, bba);
    float q2 = 2.0 * dot(pos * ray_dir, bba);
    float q3 = dot(pos * pos, bba) - a * b;

    float p = q2 / q1;
    float q = q3 / q1;

    float tmp = 0.25 * p * p - q;
    if (tmp >= 0.0) {
        float tmp_sqrt = sqrt(tmp);
        return vec3(-0.5 * p - tmp_sqrt, -0.5 * p + tmp_sqrt, 1.0);
    } else {
        return vec3(0.0);
    }
}


vec2 distort(vec2 pixel_coords) {
    float r2 = dot(pixel_coords, pixel_coords);
    float dr = 1.0 + k1 * r2 + k2 * r2 * r2;
    return pixel_coords * dr;
}


vec3 get_surface_normal(vec3 pos) {
    float a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS;
    float b = JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS;
    return pos * vec3(a, a, b);
}


ivec2 reshape(ivec2 coords) {
    int total_coord = coords.x + coords.y * 1648;
    int uebertrag = total_coord / 26368;
    return ivec2(total_coord - uebertrag * 26368, uebertrag);
}

vec2 get_sub_stripe_color(vec2 pixel_coords, ivec2 stripe_offset) {
    // non-photoactive pixels are excluded from the total strip of width 1630
    // also pixels whose edge is too close to the edge of the stripe are excluded
    ivec2 extent_begin = ivec2(24, 0.0);
    ivec2 extent_end = ivec2(1630, 127.0);

    if (pixel_coords.x < extent_begin.x || pixel_coords.x > extent_end.x || pixel_coords.y < extent_begin.y || pixel_coords.y > extent_end.y) {
        return vec2(0.0);
    }

    vec2 base_pixel_vec = floor(pixel_coords);
    vec2 offset = pixel_coords - base_pixel_vec;
    ivec2 base_pixel = ivec2(base_pixel_vec);
    ivec2 color_coords;
    float color = 0.0;
    
    color_coords = reshape(clamp(base_pixel, extent_begin, extent_end)) + stripe_offset;
    color += (1.0 - offset.x) * (1.0 - offset.y) * texelFetch(input_raw_image, color_coords, 0).x;

    color_coords = reshape(clamp(base_pixel + ivec2(1, 0), extent_begin, extent_end)) + stripe_offset;
    color += offset.x * (1.0 - offset.y) * texelFetch(input_raw_image, color_coords, 0).x;

    color_coords = reshape(clamp(base_pixel + ivec2(0, 1), extent_begin, extent_end)) + stripe_offset;
    color += (1.0 - offset.x) * offset.y * texelFetch(input_raw_image, color_coords, 0).x;

    color_coords = reshape(clamp(base_pixel + ivec2(1, 1), extent_begin, extent_end)) + stripe_offset;
    color += offset.x * offset.y * texelFetch(input_raw_image, color_coords, 0).x;

    return vec2(color * color, 1.0);
}


float get_bdrf_brightness(float sun_dot, float view_dot) {
    sun_dot = (sun_dot * 511.0 + 0.5) / 512.0;
    view_dot = (view_dot * 511.0 + 0.5) / 512.0;
    return texture(bdrf_tex, vec2(sun_dot, view_dot)).x;
}


float get_height(vec3 pos) {
    vec3 new_pos = pos * vec3(1.0, 1.0, JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS);
    vec3 surf_point = normalize(new_pos) * JUPITER_EQUATORIAL_RADIUS;
    surf_point.z *= JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS;
    return length(pos - surf_point);
}


float get_closest_approach_s(vec3 pos, vec3 ray_dir) {
    pos.z *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS;
    ray_dir.z *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS;
    return -dot(pos, ray_dir) / dot(ray_dir, ray_dir);
}


float get_sun_height(vec3 pos) {
    float s = get_closest_approach_s(pos, sun_direction);
    pos.z *= JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS;
    vec3 sun_direction2 = sun_direction * vec3(1.0, 1.0, JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS);
    vec3 pos2 = pos + s * sun_direction2;
    if (s <= 0) {
        return length(pos) - JUPITER_EQUATORIAL_RADIUS;
    } else {
        return length(pos2) - JUPITER_EQUATORIAL_RADIUS;
    }
}


vec4 get_fog_surf_intersection(vec3 cam_pos, vec3 ray_dir, vec3 intersection) {
    ray_dir = normalize(ray_dir);
    vec3 surf_normal = normalize(get_surface_normal(intersection.xyz));

    float s = max(get_ball_intersection_s(cam_pos, ray_dir, cutoff_factor * atm_lambda).x, 0.0);
    vec3 point_b = cam_pos + s * ray_dir;
    float dist_in_atm = length(intersection.xyz - point_b) / float(num_steps);

    vec3 pos;
    float height, dens, alpha, sun_brightness, sun_height, dist_factor;
    float last_dens = atm_base_density;
    float last_sun_height = get_sun_height(intersection);
    vec4 total_color = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i=1; i<num_steps; i++) {
        pos = intersection - float(i) * ray_dir * dist_in_atm;
        height = get_height(pos);
        dens = atm_base_density * exp(-height / atm_lambda);
        alpha = 1.0 - exp(-(dens * 0.5 + last_dens * 0.5) * dist_in_atm);
        last_dens = dens;

        sun_height = get_sun_height(pos);
        dist_factor = clamp(sun_height / (sun_height - last_sun_height), 0.0, 1.0);
        sun_brightness = mix(step(-1.0, last_sun_height), step(-1.0, sun_height), 0.5 * (dist_factor * dist_factor + dist_factor));
        last_sun_height = sun_height;

        total_color.xyz = mix(total_color.xyz, fog_base_color * sun_brightness, alpha);
        total_color.w *= 1.0 - alpha;
    }
    total_color.w = (1.0 - total_color.w);
    total_color.xyz /= total_color.w;
    if (atm_reduced) {
        total_color.w *= 1.0 - smoothstep(0.0, 0.4, -dot(ray_dir, surf_normal));
        total_color.w *= smoothstep(-0.1, 0.2, dot(normalize(sun_direction), surf_normal));
    }
    return total_color;
}


float magic_mixing_factor(float mixing_strength, vec3 y_vector, vec3 ray_dir, vec3 surf_normal, vec2 pixel_coords) {
    float b = clamp(-dot(surf_normal, y_vector), -0.9, 0.9) * 0.75;
    float a = pixel_coords.x / 827.0;
    float kaka = a / (b + 1.0) + 1.0 - 1.0 / (b + 1.0);
    kaka = max(kaka, -a / (-b + 1.0) + 1.0 - 1.0 / (-b + 1.0));
    kaka = 1.0 - (max(kaka, mixing_strength) - mixing_strength) / (1.0 - mixing_strength);
    return -dot(surf_normal, normalize(ray_dir)) * kaka;
}


vec4 get_intersection_point_color(vec3 pos) {
    mat4x4 cam_info;
    vec3 cam_pos, ray_dir, screen_space_pos;
    mat3x3 inv_cam_mat;
    vec2 pixel_coords;

    vec3 surface_normal = normalize(get_surface_normal(pos));

    vec2 framelet_color_r;
    vec2 framelet_color_g;
    vec2 framelet_color_b;
    vec2 viewing_angle; 

    vec2 cur_color_r, cur_color_g, cur_color_b;
    vec3 color = vec3(0.0);
    vec3 color_without_edges = vec3(0.0);
    float alpha = 0.0;
    float alpha_without_edges = 0.0;

    int num_image_stripes;
    int start_counter = 0;

    float mixing_exponent = 10.0 * mixing_strength.x / ((1.0 + 1e-1 - mixing_strength.x) * (1.0 + 1e-1 - mixing_strength.x));
    vec2 cur_max_exponent = vec2(0.0);
    vec2 mixing_weight;
    float renormalization_factor;
    bool at_least_one_full_color = false;
    float test;

    float brightness;
    vec3 sun_dir;

    for (int k=0; k<num_images; k++) {
        framelet_color_r = vec2(0.0);
        framelet_color_g = vec2(0.0);
        framelet_color_b = vec2(0.0);
        viewing_angle = vec2(0.0);
        brightness = 0.0;

        num_image_stripes = image_index_buffer.table[k];
        for (int i=start_counter; i<start_counter + num_image_stripes; i++) {
            cam_info = junocam_view_buffer.table[i];
            cam_pos = cam_info[0].xyz;
            inv_cam_mat = mat3x3(cam_info[1].xyz, cam_info[2].xyz, cam_info[3].xyz);
            ray_dir = pos - cam_pos;
            sun_dir = vec3(cam_info[1].w, cam_info[2].w, cam_info[3].w);

            vec4 fog_color = get_fog_surf_intersection(cam_pos, ray_dir, pos);

            if (cam_info[0].w < 0.5) {
                continue;
            }

            if (dot(ray_dir, surface_normal) >= 0.0) {
                continue;
            }

            screen_space_pos = inv_cam_mat * ray_dir;
            if (screen_space_pos.z <= 0) {
                continue;
            }
            pixel_coords = screen_space_pos.xy * focal_length / screen_space_pos.z;
            pixel_coords = distort(pixel_coords);

            cur_color_r = get_sub_stripe_color(pixel_coords + color_centers[2], ivec2(0, i * 24 + 16)); // 256
            cur_color_g = get_sub_stripe_color(pixel_coords + color_centers[1], ivec2(0, i * 24 + 8)); // 128
            cur_color_b = get_sub_stripe_color(pixel_coords + color_centers[0], ivec2(0, i * 24));

            framelet_color_r += cur_color_r;
            framelet_color_g += cur_color_g;
            framelet_color_b += cur_color_b;

            if (cur_color_r.y > 0.0 || cur_color_g.y > 0.0 || cur_color_b.y > 0.0) {
                //brightness += get_bdrf_brightness(dot(normalize(sun_dir), surface_normal), -dot(normalize(ray_dir), surface_normal));
                viewing_angle += vec2(magic_mixing_factor(mixing_strength.y, transpose(inv_cam_mat)[0], ray_dir, surface_normal, pixel_coords), 1.0);
            }
        }
        start_counter += num_image_stripes;

        if (framelet_color_r.y > 0) {framelet_color_r.x /= framelet_color_r.y;}
        if (framelet_color_g.y > 0) {framelet_color_g.x /= framelet_color_g.y;}
        if (framelet_color_b.y > 0) {framelet_color_b.x /= framelet_color_b.y;}
        if (viewing_angle.y > 0) {viewing_angle.x /= viewing_angle.y; brightness /= viewing_angle.y;}

        //framelet_color_r.x /= brightness;
        //framelet_color_g.x /= brightness;
        //framelet_color_b.x /= brightness;



        if (framelet_color_r.y > 0.0 || framelet_color_g.y > 0.0 || framelet_color_b.y > 0.0) {
            if (viewing_angle.x > cur_max_exponent.x) {
                renormalization_factor = exp(mixing_exponent * (cur_max_exponent.x - viewing_angle.x));
                color *= renormalization_factor;
                alpha *= renormalization_factor;
                cur_max_exponent.x = viewing_angle.x;
            }
            mixing_weight.x = exp(mixing_exponent * (viewing_angle.x - cur_max_exponent.x));

            color += vec3(framelet_color_r.x, framelet_color_g.x, framelet_color_b.x) * mixing_weight.x;
            alpha += mixing_weight.x;
        }


        if (framelet_color_r.y > 0.0 && framelet_color_g.y > 0.0 && framelet_color_b.y > 0.0) {
            at_least_one_full_color = true;

            if (viewing_angle.x > cur_max_exponent.y) {
                renormalization_factor = exp(mixing_exponent * (cur_max_exponent.y - viewing_angle.x));
                color_without_edges *= renormalization_factor;
                alpha_without_edges *= renormalization_factor;
                cur_max_exponent.y = viewing_angle.x;
            }
            mixing_weight.y = exp(mixing_exponent * (viewing_angle.x - cur_max_exponent.y));

            color_without_edges += vec3(framelet_color_r.x, framelet_color_g.x, framelet_color_b.x) * mixing_weight.y;
            alpha_without_edges += mixing_weight.y;
        }

        
    }

    
    if (alpha > 0.0) {
        if (at_least_one_full_color) {
            color = color_without_edges / alpha_without_edges;
        } else {
            color = color / alpha;
        }

        alpha = 1.0;
    }

    return vec4(color, alpha);
}



vec4 get_ellipsoid_color(vec3 pos) {
    vec3 surface_normal = get_surface_normal(pos);
    float sun_brightness = smoothstep(-0.1, 0.2, dot(normalize(surface_normal), normalize(sun_direction)));
    sun_brightness = sun_brightness * 0.85 + 0.15;
    return vec4(fog_base_color * sun_brightness, 1.0); // vec3(200.0 / 255, 110.0, 23.0)
}


vec4 get_fog_no_intersection(vec3 ray_dir) {
    ray_dir = normalize(ray_dir);

    vec3 s = get_ball_intersection_s(camera_pos, ray_dir, cutoff_factor * atm_lambda);

    if (s.z == 0.0 || s.y < 0.0) {
        return vec4(0.0);
    }
    
    float step_size = (s.y - max(s.x, 0.0)) / float(num_steps);
    vec3 point_b = camera_pos + s.y * ray_dir;
    vec3 surf_normal = get_surface_normal(point_b) + get_surface_normal(point_b - float(num_steps) * step_size * ray_dir);
    
    vec3 pos;
    float height, dens, alpha, sun_brightness, sun_height, dist_factor;
    float last_dens = atm_base_density * exp(-get_height(point_b) / atm_lambda);
    float last_sun_height = get_sun_height(point_b);
    vec4 total_color = vec4(0.0, 0.0, 0.0, 1.0);
    for (int i=1; i<num_steps; i++) {
        pos = point_b - float(i) * ray_dir * step_size;
        height = get_height(pos);
        dens = atm_base_density * exp(-height / atm_lambda);
        alpha = 1.0 - exp(-(dens * 0.5 + last_dens * 0.5) * step_size);
        last_dens = dens;

        sun_height = get_sun_height(pos);
        dist_factor = clamp(sun_height / (sun_height - last_sun_height), 0.0, 1.0);
        sun_brightness = mix(step(-1.0, last_sun_height), step(-1.0, sun_height), 0.5 * (dist_factor * dist_factor + dist_factor));
        last_sun_height = sun_height;

        total_color.xyz = mix(total_color.xyz, fog_base_color * sun_brightness, alpha);
        total_color.w *= 1.0 - alpha;
    }
    total_color.w = (1.0 - total_color.w);
    total_color.xyz /= total_color.w;
    if (atm_reduced) {
        total_color.w *= smoothstep(-0.1, 0.2, dot(normalize(sun_direction), normalize(surf_normal)));
    }
    return total_color;
}


void main() {
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    if (pixel_coords.x < resolution.x && pixel_coords.y < resolution.y) {

    vec2 tex_coords = (vec2(pixel_coords) + 0.5 - 0.5 * vec2(resolution)) / float(max(resolution.x, resolution.y));

    vec3 ray_dir = camera_mat * vec3(tex_coords, 1.0);

    vec4 intersection = get_intersection_point(camera_pos, ray_dir);
    
    vec4 color_image = vec4(0.0);
    vec4 color_ellipsoid = vec4(0.0);
    vec4 fog_color;
    if (intersection.w != 0.0) {
        color_image = get_intersection_point_color(intersection.xyz);
        color_ellipsoid = get_ellipsoid_color(intersection.xyz);
        fog_color = get_fog_surf_intersection(camera_pos, ray_dir, intersection.xyz);
    } else {
        fog_color = get_fog_no_intersection(ray_dir);
    }


    imageStore(output_image_tex, pixel_coords, color_image);
    imageStore(output_ellipsoid_tex, pixel_coords, color_ellipsoid);
    imageStore(output_fog_tex, pixel_coords, fog_color);
    }
}