#type vertex
#version 450 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_tex_coords;
layout (location = 2) in float v_tex_id;

out vec2 f_tex_coords;
out float f_tex_id;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(v_position, 5.0, 1.0);
    f_tex_coords = v_tex_coords;
    f_tex_id = v_tex_id;
}

#type fragment
#version 450 core

in vec2 f_tex_coords;
in float f_tex_id;

out vec4 color;

uniform sampler2DArray uTexture;

void main()
{
    vec3 textColor = vec3(0.0, 0.0, 1.0);
    float r = texture(uTexture, vec3(f_tex_coords.xy, f_tex_id)).r;
    vec4 sampled = vec4(1.0, 1.0, 1.0, r);
    if (r > 0) color = vec4(textColor, 1.0) * sampled;
    else discard;
}
