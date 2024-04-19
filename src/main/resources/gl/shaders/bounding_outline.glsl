#type vertex
#version 330 core
layout (location = 0) in vec2 aPos;

out vec3 fColor;

uniform mat4 uVP;

void main()
{
    fColor = vec3(1, 1, 1);
    gl_Position = uVP * vec4(aPos, 5.0, 1.0);
}

#type fragment
#version 330 core
in vec3 fColor;
out vec4 color;

void main()
{
    color = vec4(fColor, .5);
}
