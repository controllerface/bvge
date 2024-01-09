#type vertex
#version 330 core
layout (location = 0) in vec2 v_position;
layout (location = 1) in vec2 v_tex_coords;
layout (location = 2) in vec4 v_transform;

out vec4 f_color;
out vec2 f_tex_coords;

uniform mat4 uVP;

vec2 rotate(vec2 vec, float angleDeg, vec2 origin)
{
    vec2 result;

    float x = vec.x - origin.x;
    float y = vec.y - origin.y;

    float cos = cos(angleDeg);
    float sin = sin(angleDeg);

    float xPrime = (x * cos) - (y * sin);
    float yPrime = (x * sin) + (y * cos);

    xPrime += origin.x;
    yPrime += origin.y;

    result.x = xPrime;
    result.y = yPrime;

    return result;
}

void main()
{
    f_tex_coords = v_tex_coords;
    vec2 pos_offset;
    pos_offset.x = v_transform.x;
    pos_offset.y = v_transform.y;
    vec2 scaled = v_position * (v_transform.w * 100);
    vec2 translated = scaled + pos_offset;
    vec2 rotated = rotate(translated, v_transform.z, pos_offset);
    gl_Position = uVP * vec4(rotated, 0.0, 1.0);
}

#type fragment
#version 330 core
in vec2 f_tex_coords;

out vec4 color;

uniform sampler2D uTextures[1];

void main()
{
    vec4 texColor = texture(uTextures[0], f_tex_coords);
    color = texColor;
}
