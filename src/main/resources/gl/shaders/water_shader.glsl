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
out vec2 FragPos;
out vec3 Normal;

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
    trs *= 2.5 * clr.a + 1.5;

    vec2 scaled1 = pos1 * trs;
    vec2 translated1 = scaled1 + pos_offset;
    fPosition = pos1 * 2;
    f_color = clr;
    FragPos = position;
    Normal = vec3(0.0, 0.0, 1.0);
    gl_Position = uVP * vec4(translated1, 3.0, 1.0);  
    EmitVertex();   

    vec2 scaled2 = pos2 * trs;
    vec2 translated2 = scaled2 + pos_offset;
    fPosition = pos2 * 2;
    f_color = clr;
    FragPos = position;
    Normal = vec3(0.0, 0.0, 1.0);
    gl_Position = uVP * vec4(translated2, 3.0, 1.0);
    EmitVertex();

    vec2 scaled3 = pos3 * trs;
    vec2 translated3 = scaled3 + pos_offset;
    fPosition = pos3 * 2;
    f_color = clr;
    FragPos = position;
    Normal = vec3(0.0, 0.0, 1.0);
    gl_Position = uVP * vec4(translated3, 3.0, 1.0);
    EmitVertex();

    vec2 scaled4 = pos4 * trs;
    vec2 translated4 = scaled4 + pos_offset;
    fPosition = pos4  * 2;
    f_color = clr;
    FragPos = position;
    Normal = vec3(0.0, 0.0, 1.0);
    gl_Position = uVP * vec4(translated4, 3.0, 1.0);
    EmitVertex();

    EndPrimitive();
}

#type fragment
#version 330 core

in vec2 fPosition;
in vec4 f_color;
in vec2 FragPos; 
in vec3 Normal;
out vec4 color;

uniform vec2 uMouse;
uniform vec2 uCamera;

void main()
{
    // diffuse
    vec3 light_position = vec3(uMouse.xy, 10.0);
    float radius = 200.0;
    vec3 normal = normalize(Normal);
    vec3 light_color = vec3(1.0, 1.0, 1.0); 
    vec3 frag_position = vec3(FragPos, 0);
    vec3 light_vector = light_position - frag_position;
    float l_distance = length(light_vector);
    vec3 light_direction = normalize(light_vector);
    float diff_mag = max(0.0, dot(light_direction, normal));
    float attenuation = pow(smoothstep(radius, 0.0, l_distance), 2);
    vec3 diffuse = diff_mag * light_color * attenuation;

    // specular
    vec3 view_source = vec3(uCamera, 100);
    vec3 view_vector = normalize(view_source - frag_position);
    vec3 reflect_vector = reflect(-light_direction, normal);
    float specular_strength = 0.5;
    float shininess = 32.0;
    float spec = pow(max(dot(view_vector, reflect_vector), 0.0), shininess);
    vec3 specular = specular_strength * spec * light_color * attenuation;

    // Ambient light
    vec3 ambient = vec3(0.31, 0.31, 0.31);

    vec3 lighting = ambient + diffuse + specular;

    float distance = 1.0 - length(fPosition);
    float circle = smoothstep(0.0, 1.0, distance);

    vec4 scaled_rgba = f_color;
    scaled_rgba.rgb *= lighting;

    if (circle > 0) color = scaled_rgba;
    else discard;
}