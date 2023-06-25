package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;

public record FBounds2D(int index)
{
    /*
    * Memory layout: float8
    * 0: x position
    * 1: y position
    * 2: width
    * 3: height
    * 4: max x value
    * 5: max y value
    * 6: [empty]
    * 7: [empty]
    *  */
    private static final int X_OFFSET = 0;
    private static final int Y_OFFSET = 1;
    private static final int W_OFFSET = 2;
    private static final int H_OFFSET = 3;
    private static final int MAX_X_OFFSET = 4;
    private static final int MAX_Y_OFFSET = 5;

    public float x()
    {
        return Main.Memory.bounds_buffer[index() + X_OFFSET];
    }

    public float y()
    {
        return Main.Memory.bounds_buffer[index() + Y_OFFSET];
    }

    public float w()
    {
        return Main.Memory.bounds_buffer[index() + W_OFFSET];
    }

    public float h()
    {
        return Main.Memory.bounds_buffer[index() + H_OFFSET];
    }

    public float min_x()
    {
        return Main.Memory.bounds_buffer[index() + X_OFFSET];
    } // min x == x

    public float min_y()
    {
        return Main.Memory.bounds_buffer[index() + Y_OFFSET];
    } // min y == y

    public float max_x()
    {
        return Main.Memory.bounds_buffer[index() + MAX_X_OFFSET];
    }

    public float max_y()
    {
        return Main.Memory.bounds_buffer[index() + MAX_Y_OFFSET];
    }
}
