#version 430


in vec3 color;
in float ray_surface_factor;

out vec4 f_color;


void main() {
    if (ray_surface_factor >= 1.0) {
        f_color = vec4(color, 1.0);
    }
}