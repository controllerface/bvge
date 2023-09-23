#type vertex
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aFlags;

out vec4 fColor;

uniform mat4 uView;
uniform mat4 uProjection;

void main()
{
    fColor = vec4(0, 0, 0, aFlags);
    gl_Position = uProjection * uView * vec4(aPos, 0.0, 1.0);
}

#type fragment
#version 330 core
in vec4 fColor;
out vec4 color;

void main()
{
    color = fColor;
}
