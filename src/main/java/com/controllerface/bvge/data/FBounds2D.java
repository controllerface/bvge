package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;

public record FBounds2D(int index) implements GameComponent
{
    /*
    * Memory layout: float16
    * 0: x position
    * 1: y position
    * 2: width
    * 3: height
    * 4: key bank offset
    * 5: spatial index key bank size
    * 6: spatial index min x offset (int cast)
    * 7: spatial index max x offset (int cast)
    * 8: spatial index min y offset (int cast)
    * 9: spatial index max y offset (int cast)
    * 10:
    * 11:
    * 12:
    * 13:
    * 14:
    * 15:
    *  */
    public static final int X_OFFSET            = 0;
    public static final int Y_OFFSET            = 1;
    public static final int W_OFFSET            = 2;
    public static final int H_OFFSET            = 3;
    public static final int BANK_OFFSET         = 4;
    public static final int SI_BANK_SIZE_OFFSET = 5;
    public static final int SI_MIN_X_OFFSET     = 6;
    public static final int SI_MAX_X_OFFSET     = 7;
    public static final int SI_MIN_Y_OFFSET     = 8;
    public static final int SI_MAX_Y_OFFSET     = 9;


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

    public int si_min_x()
    {
        return (int)Main.Memory.bounds_buffer[index() + SI_MIN_X_OFFSET];
    }

    public int si_max_x()
    {
        return (int)Main.Memory.bounds_buffer[index() + SI_MAX_X_OFFSET];
    }

    public int si_min_y()
    {
        return (int)Main.Memory.bounds_buffer[index() + SI_MIN_Y_OFFSET];
    }

    public int si_max_y()
    {
        return (int)Main.Memory.bounds_buffer[index() + SI_MAX_Y_OFFSET];
    }

    public int si_bank_size()
    {
        return (int)Main.Memory.bounds_buffer[index() + SI_BANK_SIZE_OFFSET];
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
