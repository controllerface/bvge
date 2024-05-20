package com.controllerface.bvge.physics;

/**
 * Container class for the runtime values of a uniform grid spatial partition. The grid boundary is
 * dynamically resizable and is calculated relative to the screen dimensions.
 */
public class UniformGrid
{
    public final static float SECTOR_SIZE = 1000.0f;
    public final float perimeter_width;// = 2048f;
    public final float perimeter_height;// = 1024f;
    public final float width;// = (8192f - perimeter_width) * 2;
    public final float height;// = (4096f - perimeter_height) * 2;
    public final float inner_width;// = width - perimeter_width;
    public final float inner_height;// = height - perimeter_height;
    public final float outer_width;// = width - perimeter_width;
    public final float outer_height;// = height - perimeter_height;
    public final int x_subdivisions;// = 200;
    public final int y_subdivisions;// = 100;
    public final int directory_length;// = x_subdivisions * y_subdivisions;
    public final float x_spacing;// = width / x_subdivisions;
    public final float y_spacing;// = height / y_subdivisions;

    private float x_origin = 0;
    private float y_origin = 0;
    private float inner_x_origin = 0;
    private float inner_y_origin = 0;
    int key_bank_size = 0;
    int key_map_size = 0;

    public UniformGrid(int screen_width, int screen_height)
    {
        float x = (float)screen_width * 2.5f;
        float y = (float)screen_height * 2.5f;
        System.out.println("x:" + screen_width + " y:" + screen_height);
        perimeter_width = screen_width * .20f;
        perimeter_height = screen_height * .30f;
        inner_width = x;
        inner_height = y;
        width = inner_width + perimeter_width;
        height = inner_height + perimeter_height;
        outer_width = width + perimeter_width * 2;
        outer_height = height + perimeter_height * 2;
        x_subdivisions = 200;
        y_subdivisions = 100;
        directory_length = x_subdivisions * y_subdivisions;
        x_spacing = width / x_subdivisions;
        y_spacing = height / y_subdivisions;
    }

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
