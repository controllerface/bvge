#type vertex
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in float aFlags;

out vec4 fColor;

uniform mat4 uView;
uniform mat4 uProjection;

// note: edge vertices are in world space so do not need a local transform
void main()
{
    if (aFlags == 0.0)
    {
        fColor = vec4(0, 0, 0, 1);
    }
    else
    {
        fColor = vec4(0, 0, 0, .2);
    }
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
