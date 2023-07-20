inline int calculate_key_index(int x_subdivisions, int x, int y)
{
    int key_index = x_subdivisions * y + x;
    return key_index;
}
