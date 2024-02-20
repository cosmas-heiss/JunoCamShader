#version 430

in vec3 vert_pos;
in float cam_or_mesh;

out vec3 color;
out float ray_surface_factor;

uniform float JUPITER_EQUATORIAL_RADIUS;
uniform float JUPITER_POLAR_RADIUS;
uniform mat3x3 inv_camera_mat;
uniform vec3 cam_pos;
uniform bool show_cam;
uniform bool show_ellipsoid;
uniform float aspect_ratio;


vec3 get_surface_normal(vec3 pos) {
    float a = JUPITER_POLAR_RADIUS / JUPITER_EQUATORIAL_RADIUS;
    float b = JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS;
    return pos * vec3(a, a, b);
}



void main() {
    vec3 pos_3d;
    float depth;
    if (cam_or_mesh < 0.5) {
        pos_3d = vert_pos * vec3(vec2(JUPITER_EQUATORIAL_RADIUS), JUPITER_POLAR_RADIUS);

        vec3 normal = get_surface_normal(pos_3d);

        float a = smoothstep(0.0, 0.1, dot(normalize(normal), normalize(cam_pos - pos_3d)));
        color = vec3(0.0, a, step(a, 0.0));
        ray_surface_factor = 2.0;
    } else if (show_cam) {
        pos_3d = vert_pos;

        if (show_ellipsoid) {
            vec3 ray_a = cam_pos * vec3(1.0, 1.0, JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS);
            vec3 ray_b = vert_pos * vec3(1.0, 1.0, JUPITER_EQUATORIAL_RADIUS / JUPITER_POLAR_RADIUS);
            vec3 v = ray_b - ray_a;

            float s = -dot(v, ray_a) / dot(v, v);
            s = clamp(s, 0.0, 1.0);
            ray_surface_factor = length(ray_a + s * v) / JUPITER_EQUATORIAL_RADIUS;
        } else {
            ray_surface_factor = 2.0;
        }

        color = vec3(1.0, 0.0, 0.0);
    }

    vec3 screen_space_pos = inv_camera_mat * (pos_3d - cam_pos);
    screen_space_pos.xy /= screen_space_pos.z;
    if (screen_space_pos.z <= 0.0) {
        depth = 2.0;
    } else {
        depth = tanh(screen_space_pos.z * 1e-7) * 0.99;
    }
    vec2 scaling = vec2(max(1.0 / aspect_ratio, 1.0), max(aspect_ratio, 1.0));
    gl_Position = vec4(screen_space_pos.xy * 2.0 * scaling, depth, 1.0);
}