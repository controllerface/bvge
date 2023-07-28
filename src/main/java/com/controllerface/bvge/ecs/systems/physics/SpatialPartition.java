package com.controllerface.bvge.ecs.systems.physics;

public class SpatialPartition
{
    private float width = 0;
    private float height = 0;

    // note: sub-divisions should always be divisible by 2
    private int x_subdivisions = 120;
    private int y_subdivisions = 120;
    private int directoryLength;
    private float x_spacing = 0;
    private float y_spacing = 0;
    private float x_origin = 0;
    private float y_origin = 0;
    int key_bank_size = 0;
    int key_map_size = 0;

    public SpatialPartition()
    {
        init();
    }

    // todo: partitioning needs to change a bit, instead of specifying subdivisions, the spacing is what
    //  should be static, so the resize operation will keep the space cell size but make the tracking area
    //  alone bigger or smaller. right now, resize changes the cell size which is not ideal.
    void init()
    {
        x_spacing = width / x_subdivisions;
        y_spacing = height / y_subdivisions;
        directoryLength = x_subdivisions * y_subdivisions;
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


    public int getDirectoryLength()
    {
        return directoryLength;
    }

    public int getKey_bank_size()
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
