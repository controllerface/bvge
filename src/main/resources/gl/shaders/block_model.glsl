#type vertex
#version 450 core
layout (location = 0) in vec4 v_position;
layout (location = 1) in vec2 v_tex_coords;
layout (location = 2) in vec4 v_color;
layout (location = 3) in float v_tex_slot;

out vec2 f_tex_coords;
out vec4 f_color;
out vec2 FragPos;
out vec3 Normal;
out float s_spec;
flat out int f_tex_slot;

uniform mat4 uVP;

void main()
{
    // todo: replcae this with specular map texture
    float s_buf = 0.1;
    if (v_tex_coords.x == 0.0 && v_tex_coords.y == 0.03125) s_buf = 0.5;
    if (v_tex_coords.x == 0.03125 && v_tex_coords.y == 0.03125) s_buf = 0.5;
    if (v_tex_coords.x == 0.03125 && v_tex_coords.y == 0.0625) s_buf = 0.5;
    if (v_tex_coords.x == 0.0 && v_tex_coords.y == 0.0625) s_buf = 0.5;
    s_spec = s_buf;

    f_tex_coords = v_tex_coords;
    f_tex_slot = int(v_tex_slot);
    f_color = v_color;
    FragPos = v_position.xy;
    Normal = vec3(0, 0, 1.0);
    gl_Position = uVP * v_position;
}

#type fragment
#version 450 core

in vec2 f_tex_coords;
in vec4 f_color;
in vec2 FragPos; 
in vec3 Normal;
in float s_spec;
flat in int f_tex_slot;

out vec4 color;

uniform sampler2D uTextures[16];
uniform vec2 uMouse;
uniform vec2 uCamera;

float calculateAmbientLight(float y) 
{
    const float minLight = 0.01;
    const float maxLight = 0.7;
    float lightValue;

    if (y < 0.0) 
    {
        lightValue = minLight;
    } 
    else if (y >= 500.0) 
    {
        lightValue = maxLight;
    } 
    else 
    {
        lightValue = mix(minLight, maxLight, y / 500.0);
    }
    return lightValue;
}


void main()
{
    // diffuse
    vec3 light_position = vec3(uMouse.xy, 10.0);
    float radius = 250.0;
    vec3 normal = normalize(Normal);
    vec3 light_color = vec3(1.0, 1.0, 1.0); 
    vec3 light_color2 = vec3(0.7, 0.5, 0.5); 
    vec3 frag_position = vec3(FragPos, 0);
    vec3 light_vector = light_position - frag_position;
    float distance = length(light_vector);
    vec3 light_direction = normalize(light_vector);
    float diff_mag = max(0.0, dot(light_direction, normal));
    float attenuation = pow(smoothstep(radius, 0.0, distance), .8);
    vec3 diffuse = diff_mag * light_color * attenuation;

    // specular
    vec3 view_source = vec3(uCamera, 100);
    vec3 view_vector = normalize(view_source - frag_position);
    vec3 reflect_vector = reflect(-light_direction, normal);
    float specular_strength =  s_spec;
    float shininess = 64.0;
    float spec = pow(max(dot(view_vector, reflect_vector), 0.0), shininess);
    vec3 specular = specular_strength * spec * light_color * attenuation;

    // Ambient light
    float l = calculateAmbientLight(FragPos.y);
    vec3 ambient = vec3(l, l, l);

    vec3 lighting = ambient + diffuse + specular;

    // Sample the texture color
    vec4 texColor = texture(uTextures[f_tex_slot], f_tex_coords);
    vec3 scaledRGB = texColor.rgb * f_color.rgb;
    scaledRGB *= lighting;

    // Output the final color
    color = vec4(scaledRGB, texColor.a);
}
