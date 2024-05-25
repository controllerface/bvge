#type vertex
#version 330 core
layout (location = 0) in vec4 aTransform;
layout (location = 1) in vec4 v_color;

uniform mat4 uVP;

out VertexData 
{
    vec4 transform;
    vec4 color;
} vertex_data;

void main()
{
    vec2 pos_offset = aTransform.xy;
    gl_Position = uVP * vec4(pos_offset, 3.0 + aTransform.z , 1.0);
    vertex_data.transform = aTransform;
    vertex_data.color = v_color;
}

#type geometry
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

uniform mat4 uVP;

in VertexData 
{
    vec4 transform;
    vec4 color;
} vertex_data[];

out vec2 fPosition;
out vec4 f_color;

void main()
{
    vec2 position = vertex_data[0].transform.xy;
    vec2 pos1 = vec2(  0.5, -0.5 );
    vec2 pos2 = vec2(  0.5,  0.5 );
    vec2 pos3 = vec2( -0.5, -0.5 );
    vec2 pos4 = vec2( -0.5,  0.5 );

    vec4 clr = vertex_data[0].color;

    vec2 pos_offset;
    pos_offset.x = vertex_data[0].transform.x;
    pos_offset.y = vertex_data[0].transform.y;

    float trs = vertex_data[0].transform.w;
    trs *= 2 * clr.a + 2;

    vec2 scaled1 = pos1 * trs;
    vec2 translated1 = scaled1 + pos_offset;
    fPosition = pos1 * 2;
    f_color = clr;
    gl_Position = uVP * vec4(translated1, 3.0, 1.0);  
    EmitVertex();   

    vec2 scaled2 = pos2 * trs;
    vec2 translated2 = scaled2 + pos_offset;
    fPosition = pos2 * 2;
    f_color = clr;
    gl_Position = uVP * vec4(translated2, 3.0, 1.0);
    EmitVertex();

    vec2 scaled3 = pos3 * trs;
    vec2 translated3 = scaled3 + pos_offset;
    fPosition = pos3 * 2;
    f_color = clr;
    gl_Position = uVP * vec4(translated3, 3.0, 1.0);
    EmitVertex();

    vec2 scaled4 = pos4 * trs;
    vec2 translated4 = scaled4 + pos_offset;
    fPosition = pos4  * 2;
    f_color = clr;
    gl_Position = uVP * vec4(translated4, 3.0, 1.0);
    EmitVertex();

    EndPrimitive();
}

#type fragment
#version 330 core

in vec2 fPosition;
in vec4 f_color;
out vec4 color;

void main()
{
    //vec4 circleColor = vec4(0.0, 0.1, 0.2, .25);
    float thickness = 1.0;
    float fade = 0.00005;

    float distance = 1.0 - length(fPosition);
    float circle = smoothstep(0.0, fade, distance);
    circle *= smoothstep(thickness + fade, thickness, distance);

    vec4 sclaed_rgba = f_color;

    if (circle > 0) color = sclaed_rgba;
    else discard;
}