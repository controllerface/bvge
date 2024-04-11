#type vertex
#version 450 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_tex_coords;

out vec2 f_tex_coords;

uniform mat4 uVP;

void main()
{
    f_tex_coords = v_tex_coords;
    gl_Position =  uVP * vec4(v_position, 0.0, 1.0);
}

#type fragment
#version 450 core

in vec2 f_tex_coords;

out vec4 color;

uniform sampler2D uTextures[1];

void main()
{
    vec4 texColor = texture(uTextures[0], f_tex_coords);
    color = texColor * vec4(.9, .9, .9, 1);
    //color = texColor;
}
