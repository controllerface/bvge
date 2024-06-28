#type vertex
#version 450 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_tex_coords;

out vec2 f_tex_coords;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(v_position, 5.0, 1.0);
    f_tex_coords = v_tex_coords;
}

#type fragment
#version 450 core

in vec2 f_tex_coords;

out vec4 color;

uniform sampler2D uTexture;

void main()
{
    vec3 textColor = vec3(0.0, 0.0, 1.0);
    float r = texture(uTexture, f_tex_coords).r;
    vec4 sampled = vec4(1.0, 1.0, 1.0, r);
    if (r > 0) color = vec4(textColor, 1.0) * sampled;
    else discard;
}
