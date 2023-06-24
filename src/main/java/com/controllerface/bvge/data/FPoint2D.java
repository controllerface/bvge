package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import org.joml.Vector2f;
import org.joml.Vector2fc;

public record FPoint2D(int index)
{
    private static int x_offset = 0;
    private static int y_offset = 1;
    private static int px_offset = 2;
    private static int py_offset = 3;

    public float pos_x()
    {
        return Main.Memory.point_buffer[index() + x_offset];
    }

    public float pos_y()
    {
        return Main.Memory.point_buffer[index() + y_offset];
    }

    public float prv_x()
    {
        return Main.Memory.point_buffer[index() + px_offset];
    }

    public float prv_y()
    {
        return Main.Memory.point_buffer[index() + py_offset];
    }

    public void addPos(FPoint2D other, Vector2f out)
    {
        out.x = pos_x() + other.pos_x();
        out.y = pos_y() + other.pos_y();
    }

    public void subPos(FPoint2D other, Vector2f out)
    {
        out.x = pos_x() - other.pos_x();
        out.y = pos_y() - other.pos_y();
    }

    public float distance(FPoint2D other)
    {
        return Vector2f.distance(other.pos_x(), other.pos_y(), pos_x(), pos_y());
    }

    public void addPos(Vector2f other)
    {
        Main.Memory.point_buffer[index() + x_offset] = pos_x() + other.x();
        Main.Memory.point_buffer[index() + y_offset] = pos_y() + other.y();
    }

    public void subPos(Vector2f other)
    {
        Main.Memory.point_buffer[index() + x_offset] = pos_x() - other.x();
        Main.Memory.point_buffer[index() + y_offset] = pos_y() - other.y();
    }

    public float dotPos(Vector2fc v)
    {
        return pos_x() * v.x() + pos_y() * v.y();
    }

    public void frameDiff(Vector2f out)
    {
        out.x = pos_x() - prv_x();
        out.y = pos_y() - prv_y();
    }

    public void frameSwap()
    {
        Main.Memory.point_buffer[index() + px_offset] = pos_x();
        Main.Memory.point_buffer[index() + py_offset] = pos_y();
    }
}
