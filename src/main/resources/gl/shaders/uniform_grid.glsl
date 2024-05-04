#type vertex
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor;

out vec4 fColor;

uniform mat4 uVP;

void main()
{
    fColor = aColor;
    gl_Position = uVP * vec4(aPos, 5.0, 1.0);
}

#type fragment
#version 330 core
in vec4 fColor;
out vec4 color;

void main()
{
    color = fColor;
}
