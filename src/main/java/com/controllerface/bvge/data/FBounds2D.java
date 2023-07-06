package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;

public record FBounds2D(int index) implements GameComponent
{
    /*
    * Memory layout: float8
    * 0: x position
    * 1: y position
    * 2: width
    * 3: height
    * 4: key bank offset
    * 5: bounding out count
    * 6:
    * 7:
    *  */
    private static final int X_OFFSET         = 0;
    private static final int Y_OFFSET         = 1;
    private static final int W_OFFSET         = 2;
    private static final int H_OFFSET         = 3;
    private static final int BANK_OFFSET      = 4;
    private static final int BOUND_OUT_OFFSET = 5;

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

    public float boo()
    {
        return Main.Memory.bounds_buffer[index() + BOUND_OUT_OFFSET];
    }

    public int bank_offset()
    {
        return (int) Main.Memory.bounds_buffer[index() + BANK_OFFSET];
    }

    public void setBankOffset(int offset)
    {
        Main.Memory.bounds_buffer[index() + BANK_OFFSET] = (float)offset;
    }
}
