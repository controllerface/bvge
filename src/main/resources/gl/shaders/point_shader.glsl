#type vertex
#version 450 core
layout (location = 0) in vec2 v_position;

uniform mat4 uVP;

void main()
{
    gl_Position =  uVP * vec4(v_position, 0.0, 1.0);
}

#type fragment
#version 450 core

out vec4 color;

void main()
{
    color = vec4(1.0, 1.0, 1.0, 0.1);
}
