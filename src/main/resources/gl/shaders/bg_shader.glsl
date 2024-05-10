#type vertex
#version 450 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_tex_coords;

out vec2 f_tex_coords;

uniform mat4 uVP;

void main()
{
    f_tex_coords = v_tex_coords;
    gl_Position =  vec4(v_position, 0.99, 1.0);
}

#type fragment
#version 450 core

in vec2 f_tex_coords;

out vec4 color;

uniform sampler2D uTextures[1];

void main()
{
    color = texture(uTextures[0], f_tex_coords);
}
