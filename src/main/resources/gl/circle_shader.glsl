#type vertex
#version 330 core
layout (location = 0) in vec4 aTransform;

uniform mat4 uView;
uniform mat4 uProjection;

out VS_OUT {
    vec4 transform;
} vs_out;

void main()
{
    vec2 pos_offset;
    pos_offset.x = aTransform.x;
    pos_offset.y = aTransform.y;
    gl_Position = uProjection * uView * vec4(pos_offset, 0.0, 1.0);
    
    vs_out.transform = aTransform;
}

#type geometry
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 uView;
uniform mat4 uProjection;

in VS_OUT {
    vec4 transform;
} vs_in[];

void main()
{
    vec2 position = vs_in[0].transform.xy;
    vec2 pos1 = vec2(-0.5, -0.5);
    vec2 pos2 = vec2( 0.5, -0.5);
    vec2 pos3 = vec2(-0.5,  0.5);
    vec2 pos4 = vec2( 0.5,  0.5);

    vec2 pos_offset;
    pos_offset.x = vs_in[0].transform.x;
    pos_offset.y = vs_in[0].transform.y;

    vec2 scaled1 = pos1 * vs_in[0].transform.w;
    vec2 translated1 = scaled1 + pos_offset;

    vec2 scaled2 = pos2 * vs_in[0].transform.w;
    vec2 translated2 = scaled2 + pos_offset;

    vec2 scaled3 = pos3 * vs_in[0].transform.w;
    vec2 translated3 = scaled3 + pos_offset;

    vec2 scaled4 = pos4 * vs_in[0].transform.w;
    vec2 translated4 = scaled4 + pos_offset;

    gl_Position = uProjection * uView * vec4(translated1, 0.0, 1.0); // 1:bottom-left   
    EmitVertex();   
    gl_Position = uProjection * uView * vec4(translated2, 0.0, 1.0); // 2:bottom-right
    EmitVertex();
    gl_Position = uProjection * uView * vec4(translated3, 0.0, 1.0); // 3:top-left
    EmitVertex();
    gl_Position = uProjection * uView * vec4(translated4, 0.0, 1.0); // 4:top-right
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