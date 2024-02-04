package com.controllerface.bvge.physics;

/**
 * Container class for the runtime values of a uniform grid spatial partition. The grid boundary is
 * dynamically resizable and is calculated relative to the screen dimensions.
 */
public class UniformGrid
{
    private float width = 0;
    private float height = 0;

    private final int x_subdivisions = 120;
    private final int y_subdivisions = 120;
    private final int directoryLength;
    private float x_spacing = 0;
    private float y_spacing = 0;
    private float x_origin = 0;
    private float y_origin = 0;
    int key_bank_size = 0;
    int key_map_size = 0;

    public UniformGrid()
    {
        directoryLength = x_subdivisions * y_subdivisions;
        init();
    }

    void init()
    {
        x_spacing = width / x_subdivisions;
        y_spacing = height / y_subdivisions;
    }

    public void resize(float width, float height)
    {
        this.width = width;
        this.height = height;
        init();
    }

    public void updateOrigin(float x_origin, float y_origin)
    {
        this.x_origin = x_origin;
        this.y_origin = y_origin;
    }


    public void resizeBank(int size)
    {
        key_bank_size = size;
        key_map_size = size / 2;
    }


    public int get_directory_length()
    {
        return directoryLength;
    }

    public int get_key_bank_size()
    {
        return key_bank_size;
    }

    public int getKey_map_size()
    {
        return key_map_size;
    }

    public float getWidth()
    {
        return width;
    }

    public float getHeight()
    {
        return height;
    }

    public float getX_spacing()
    {
        return x_spacing;
    }

    public float getY_spacing()
    {
        return y_spacing;
    }

    public float getX_origin()
    {
        return x_origin;
    }

    public float getY_origin()
    {
        return y_origin;
    }

    public int getX_subdivisions()
    {
        return x_subdivisions;
    }

    public int getY_subdivisions()
    {
        return y_subdivisions;
    }
}
