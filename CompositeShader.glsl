#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform sampler2D jupiter_image_tex;
layout(binding = 1) uniform sampler2D ellipsoid_tex;
layout(binding = 2) uniform usampler2D wireframe_tex;
layout(binding = 3) uniform sampler2D fog_tex;
layout(binding = 4) uniform writeonly uimage2D output_tex;


uniform ivec2 resolution;
uniform bool render_atm;
uniform ivec4 wf_render_mode;
uniform float aspect_ratio;
uniform bool super_sampling;



void main() {
    ivec2 pixel_coords_base = ivec2(gl_GlobalInvocationID.xy);
    int factor = 1;
    if (super_sampling) {
        factor = 2;
    }
    if (pixel_coords_base.x < resolution.x / factor && pixel_coords_base.y < resolution.y / factor) {
        vec3 out_color = vec3(0.0);

        ivec2 pixel_coords;
        for (int i=0; i<factor; i++) {
            for (int j=0; j<factor; j++) {
                pixel_coords = pixel_coords_base * factor + ivec2(i, j);

                vec3 color = vec3(0.0);
                vec4 image_color = texelFetch(jupiter_image_tex, pixel_coords, 0);
                vec4 ellipsoid_color = texelFetch(ellipsoid_tex, pixel_coords, 0);
                float ellipsoid_step = step(1e-13, ellipsoid_color.w);
                vec3 mesh_color = texelFetch(wireframe_tex, pixel_coords / factor, 0).xyz / 255.0;
                vec4 fog_color = texelFetch(fog_tex, pixel_coords, 0);

                
                if (wf_render_mode.y == 1 && wf_render_mode.x == 0) {
                    color = mix(vec3(0.0), vec3(step(1e-13, mesh_color.z) * 0.4), ellipsoid_step);
                }

                
                if (wf_render_mode.x == 1) {
                    color = ellipsoid_color.xyz;
                }

                if (wf_render_mode.z == 1) {
                    color = mix(color, image_color.xyz, step(1e-13, image_color.w));
                }

                if (wf_render_mode.y == 1) {
                    color = mix(vec3(mesh_color.y) * 0.6 + 0.4, color, 1.0 - ellipsoid_step * step(1e-13, mesh_color.y));
                }

                if (wf_render_mode.z == 0) {
                    color = mix(color, image_color.xyz, step(1e-13, image_color.w));
                }

                if (render_atm) {
                    color = mix(color, fog_color.xyz, fog_color.w);
                }

                if (wf_render_mode.w == 1) {
                    color = mix(color, vec3(1.0, 0.0, 0.0), step(1e-13, mesh_color.x));
                }


                float size_x = min(aspect_ratio, 1.0) * resolution.x;
                float size_y = min(1.0 / aspect_ratio, 1.0) * resolution.y;
                ivec2 range_x = ivec2((float(resolution.x) - size_x) * 0.5 - 1, (float(resolution.x) + size_x) * 0.5);
                ivec2 range_y = ivec2((float(resolution.y) - size_y) * 0.5 - 1, (float(resolution.y) + size_y) * 0.5);
                
                if (pixel_coords.x == range_x.x || pixel_coords.x == range_x.y || pixel_coords.y == range_y.x || pixel_coords.y == range_y.y) {
                    color = vec3(1.0, 0.0, 0.0);
                } else if (pixel_coords.x < range_x.x || pixel_coords.x > range_x.y || pixel_coords.y < range_y.x || pixel_coords.y > range_y.y) {
                    color = mix(color, vec3(0.3, 0.3, 0.3), 0.5);
                }

                out_color += color;

                
            }
        }

        out_color = out_color * 255.0 / float(factor * factor) + 0.5;
        imageStore(output_tex, pixel_coords_base, uvec4(out_color, 0));
    }
}