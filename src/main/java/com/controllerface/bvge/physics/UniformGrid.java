package com.controllerface.bvge.physics;

/**
 * Container class for the runtime values of a uniform grid spatial partition. The grid boundary is
 * dynamically resizable and is calculated relative to the screen dimensions.
 */
public class UniformGrid
{
    public final float width = 8000 * 2;
    public final float height = 4000 * 2;
    public final float inner_width = 8000 * 1.5f;
    public final float inner_height = 4000 * 1.5f;
    public final int x_subdivisions = 200;
    public final int y_subdivisions = 100;
    public final int directory_length = x_subdivisions * y_subdivisions;
    public final float x_spacing = width / x_subdivisions;
    public final float y_spacing = height / y_subdivisions;

    private float x_origin = 0;
    private float y_origin = 0;
    private float inner_x_origin = 0;
    private float inner_y_origin = 0;
    int key_bank_size = 0;
    int key_map_size = 0;

    public UniformGrid() { }

    public void updateOrigin(float x_origin, float y_origin)
    {
        this.x_origin = x_origin;
        this.y_origin = y_origin;
        this.inner_x_origin = this.x_origin + (width - inner_width) / 2;
        this.inner_y_origin = this.y_origin + (height - inner_height) / 2;
    }

    public void resizeBank(int size)
    {
        key_bank_size = size;
        key_map_size = size / 2;
    }

    public int get_key_bank_size()
    {
        return key_bank_size;
    }

    public int getKey_map_size()
    {
        return key_map_size;
    }


    public float x_origin()
    {
        return x_origin;
    }

    public float y_origin()
    {
        return y_origin;
    }

    public float inner_x_origin()
    {
        return inner_x_origin;
    }

    public float inner_y_origin()
    {
        return inner_y_origin;
    }
}
