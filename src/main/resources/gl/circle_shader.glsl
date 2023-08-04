#type vertex
#version 330 core
layout (location = 0) in vec4 aTransform;

uniform mat4 uView;
uniform mat4 uProjection;

void main()
{
    vec2 pos_offset;
    pos_offset.x = aTransform.x;
    pos_offset.y = aTransform.y;
    gl_Position = uProjection * uView * vec4(pos_offset, 0.0, 1.0);
}

#type geometry
#version 330 core
layout (points) in;
layout (points, max_vertices = 1) out;

void main()
{
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    EndPrimitive();
}

#type fragment
#version 330 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(0.0, 1.0, 0.0, 1.0);
}