/**
Rotates a target point around an origin point, by the specified angle. 
The target point's previous positions are forwarded to the output vector.
 */
inline float4 rotate_point(float4 target, float2 origin, float angle)
{
    float4 output;

    float rad = radians(angle);

    float cosine = native_cos(rad);
    float sine = native_sin(rad);

    float x_pos = target.x - origin.x;
    float y_pos = target.y - origin.y;

    float x_prime_pos = (x_pos * cosine) - (y_pos * sine);
    float y_prime_pos = (x_pos * sine) + (y_pos * cosine);
        
    x_prime_pos += origin.x;
    y_prime_pos += origin.y;
    output.x = x_prime_pos;
    output.y = y_prime_pos;

    output.z = target.z;
    output.w = target.w;
    
    return output;
}
