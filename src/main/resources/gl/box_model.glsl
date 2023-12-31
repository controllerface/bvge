#type vertex
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aTransform;
layout (location = 2) in vec2 aTexCoords;
layout (location = 3) in vec4 aColor;

out vec4 fColor;
out vec2 fTexCoords;

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
    fColor = aColor;
    fTexCoords = aTexCoords;
    vec2 pos_offset;
    pos_offset.x = aTransform.x;
    pos_offset.y = aTransform.y;
    vec2 scaled = aPos * (aTransform.w * 100);
    vec2 translated = scaled + pos_offset;
    vec2 rotated = rotate(translated, aTransform.z, pos_offset);
    gl_Position = uVP * vec4(rotated, 0.0, 1.0);
}

#type fragment
#version 330 core
in vec4 fColor;
in vec2 fTexCoords;

out vec4 color;

uniform sampler2D uTextures[1];

void main()
{
    vec4 texColor = texture(uTextures[0], fTexCoords);
    color = fColor * texColor;
}
