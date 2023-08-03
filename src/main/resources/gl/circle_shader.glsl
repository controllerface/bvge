#type vertex
#version 330 core
layout (location = 0) in vec3 aPos;// Vertex position attribute

out vec3 fColor;

uniform mat4 uView;
uniform mat4 uProjection;

void main()
{
    fColor = vec3(0, 0, 0);
    gl_Position = uProjection * uView * vec4(aPos, 1.0);
}

#type fragment
#version 330 core
uniform vec2 iResolution;  // Screen resolution

in vec3 fColor;
out vec4 fragColor;

void main()
{
    // Parameters
    vec3 circleColor = vec3(0.85, 0.35, 0.2);
    float thickness = 0.01;
    float fade = 0.005;

    // -1 -> 1 local space, adjusted for aspect ratio
    vec2 uv = (gl_FragCoord.xy / iResolution.xy) * 2.0 - 1.0;
    float aspect = iResolution.x / iResolution.y;
    uv.x *= aspect;

    // Calculate distance and fill circle with white
    float distance = 1.0 - length(uv);
    vec3 color = vec3(smoothstep(0.0, fade, distance));
    color *= vec3(smoothstep(thickness + fade, thickness, distance));

    // Set output color
    fragColor = vec4(color * circleColor, 1.0);
}