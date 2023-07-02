package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.ecs.components.GameComponent;
import org.joml.Vector2f;

public record FTransform(int index) implements GameComponent
{
    private static int x_offset = 0;
    private static int y_offset = 1;
    private static int sx_offset = 2;
    private static int sy_offset = 3;

    public float pos_x()
    {
        return Main.Memory.body_buffer[index() + x_offset];
    }

    public float pos_y()
    {
        return Main.Memory.body_buffer[index() + y_offset];
    }

    public float scale_x()
    {
        return Main.Memory.body_buffer[index() + sx_offset];
    }

    public float scale_y()
    {
        return Main.Memory.body_buffer[index() + sy_offset];
    }
}
