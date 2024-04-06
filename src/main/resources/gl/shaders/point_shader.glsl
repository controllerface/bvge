#type vertex
#version 450 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec4 v_color;

out vec4 f_color;

uniform mat4 uVP;

void main()
{
    f_color = v_color;
    gl_Position =  uVP * vec4(v_position, 0.0, 1.0);
}

#type fragment
#version 450 core

in vec4 f_color;

out vec4 color;

void main()
{
    color = f_color;
}
