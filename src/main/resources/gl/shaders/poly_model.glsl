#type vertex
#version 450 core
layout (location = 0) in vec4 v_position;
layout (location = 1) in vec2 v_tex_coords;
layout (location = 2) in vec4 v_color;

out vec2 f_tex_coords;

uniform mat4 uVP;

void main()
{
    f_tex_coords = v_tex_coords;
    gl_Position = uVP * v_position; //vec4(v_position, v_side, 1.0);
}

#type fragment
#version 450 core

in vec2 f_tex_coords;

out vec4 color;

uniform sampler2D uTextures[1];

void main()
{
    vec4 texColor = texture(uTextures[0], f_tex_coords);
    color = texColor;
}
