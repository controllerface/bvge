#type vertex
#version 330 core
layout (location = 0) in vec4 aTransform;

uniform mat4 uVP;

out VertexData 
{
    vec4 transform;
    float screen;
} vertex_data;

void main()
{
    vertex_data.screen = aTransform.z == -1 ? 1 : 0;
    vertex_data.transform = aTransform;
    float z = aTransform.z == -1 ? 6.0 : 5.0;
    gl_Position = uVP * vec4(aTransform.xy, z, 1.0);
}

#type geometry
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 uVP;

in VertexData 
{
    vec4 transform;
    float screen;
} vertex_data[];

out vec2 fPosition;
flat out float screen;

void main()
{
    vec2 position = vertex_data[0].transform.xy;
    vec2 pos1 = vec2(  0.5, -0.5 );
    vec2 pos2 = vec2(  0.5,  0.5 );
    vec2 pos3 = vec2( -0.5, -0.5 );
    vec2 pos4 = vec2( -0.5,  0.5 );

    vec2 pos_offset;
    pos_offset.x = vertex_data[0].transform.x;
    pos_offset.y = vertex_data[0].transform.y;

    vec2 scaled1 = pos1 * vertex_data[0].transform.w;
    vec2 translated1 = scaled1 + pos_offset;
    fPosition = pos1 * 2;
    screen = vertex_data[0].screen;
    gl_Position = uVP * vec4(translated1, 5.0, 1.0);  
    EmitVertex();   

    vec2 scaled2 = pos2 * vertex_data[0].transform.w;
    vec2 translated2 = scaled2 + pos_offset;
    fPosition = pos2 * 2;
    screen = vertex_data[0].screen;
    gl_Position = uVP * vec4(translated2, 5.0, 1.0);
    EmitVertex();

    vec2 scaled3 = pos3 * vertex_data[0].transform.w;
    vec2 translated3 = scaled3 + pos_offset;
    fPosition = pos3 * 2;
    screen = vertex_data[0].screen;
    gl_Position = uVP * vec4(translated3, 5.0, 1.0);
    EmitVertex();

    vec2 scaled4 = pos4 * vertex_data[0].transform.w;
    vec2 translated4 = scaled4 + pos_offset;
    fPosition = pos4  * 2;
    screen = vertex_data[0].screen;
    gl_Position = uVP * vec4(translated4, 5.0, 1.0);
    EmitVertex();

    EndPrimitive();
}

#type fragment
#version 330 core

in vec2 fPosition;
flat in float screen;

out vec4 color;

void main()
{
    float r = screen > 0.0 ? 0.5 : 0.0;
    float g = screen > 0.0 ? 0.0 : 0.5;
    float b = 0;
    vec4 circleColor = vec4(r, g, b, .3);
    float thickness = .4;
    float fade = .0005;

    float distance = 1.0 - length(fPosition);
    float circle = smoothstep(0.0, fade, distance);
    circle *= smoothstep(thickness + fade, thickness, distance);
    
    if (circle > 0) color = circleColor;
    else discard;
}