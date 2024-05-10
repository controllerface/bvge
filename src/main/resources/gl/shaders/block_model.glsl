#type vertex
#version 450 core
layout (location = 0) in vec4 v_position;
layout (location = 1) in vec2 v_tex_coords;
layout (location = 2) in vec4 v_color;

out vec2 f_tex_coords;
out vec4 f_color;

uniform mat4 uVP;

void main()
{
    f_tex_coords = v_tex_coords;
    f_color = v_color;
    gl_Position = uVP * v_position;
}

#type fragment
#version 450 core

in vec2 f_tex_coords;
in vec4 f_color;

out vec4 color;

uniform sampler2D uTextures[1];

void main()
{
    vec4 texColor = texture(uTextures[0], f_tex_coords);
    vec3 scaledRGB = texColor.rgb * f_color.rgb;
    color = vec4(scaledRGB, texColor.a);
}
