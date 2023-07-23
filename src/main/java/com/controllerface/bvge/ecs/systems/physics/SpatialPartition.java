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
    int[] key_bank = new int[0];
    int[] key_map = new int[0];
    int[] key_counts = new int[0];
    int[] key_offsets = new int[0];

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

    private int calculateKeyIndex(int x, int y)
    {
        int key_index = x_subdivisions * y + x;
        return key_index;
    }

    public void resizeBank(int size)
    {
        key_bank_size = size;
        key_map_size = size / 2;
        key_map     = new int[key_map_size];
        key_bank    = new int[key_bank_size];
        key_counts  = new int[directoryLength];
        key_offsets = new int[directoryLength];
    }

    public int[] getKey_counts()
    {
        return key_counts;
    }

    public int[] getKey_bank()
    {
        return key_bank;
    }

    public int[] getKey_offsets()
    {
        return key_offsets;
    }

    public int[] getKey_map()
    {
        return key_map;
    }

    public float getWidth() {
        return width;
    }

    public float getHeight() {
        return height;
    }

    public float getX_spacing() {
        return x_spacing;
    }

    public float getY_spacing() {
        return y_spacing;
    }

    public float getX_origin() {
        return x_origin;
    }

    public float getY_origin() {
        return y_origin;
    }

    public int getX_subdivisions() {
        return x_subdivisions;
    }

    public int getY_subdivisions() {
        return y_subdivisions;
    }

    int[] getKeyForPoint(float px, float py)
    {
        float adjusted_x = px - (x_origin);
        float adjusted_y = py - (y_origin);
        int index_x = ((int) Math.floor(adjusted_x / x_spacing));
        int index_y = ((int) Math.floor(adjusted_y / y_spacing));
        int[] out = new int[2];
        out[0] = index_x;
        out[1] = index_y;
        return out;
    }

    public int countAtIndex(float x, float y)
    {
        if (x < x_origin || x > x_origin + width)
        {
            return 0;
        }

        int[] key = getKeyForPoint(x, y);
        int i = calculateKeyIndex(key[0], key[1]);
        if (i > key_counts.length-1)
        {
            return 0;
        }
        return key_counts[i];
    }
}
