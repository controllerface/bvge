inline float4 rotate_point(float4 target, float2 origin, float angle)
{
    float4 output;

    float rad = radians(angle);

    float cosine = cos(rad);
    float sine = sin(rad);

    float x_pos = target.x - origin.x;
    float y_pos = target.y - origin.y;
    // float x_prv = target.z - origin.x;
    // float y_prv = target.w - origin.y;

    float x_prime_pos = (x_pos * cosine) - (y_pos * sine);
    float y_prime_pos = (x_pos * sine) + (y_pos * cosine);
    // float x_prime_prv = (x_prv * cosine) - (y_prv * sine);
    // float y_prime_prv = (x_prv * sine) + (y_prv * cosine);

    x_prime_pos += origin.x;
    y_prime_pos += origin.y;
    // x_prime_prv += origin.x;
    // y_prime_prv += origin.y;

    output.x = x_prime_pos;
    output.y = y_prime_pos;
    // output.z = x_prime_prv;
    // output.w = y_prime_prv;
    output.z = target.z;
    output.w = target.w;

    return output;
}
