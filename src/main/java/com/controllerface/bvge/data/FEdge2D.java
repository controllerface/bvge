package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;

public record FEdge2D(int index, FPoint2D p1, FPoint2D p2)
{
    public int p1_index()
    {
        return (int)Main.Memory.edge_buffer[index];
    }

    public int p2_index()
    {
        return (int)Main.Memory.edge_buffer[index + 1];
    }

    public float length()
    {
        return Main.Memory.edge_buffer[index + 2];
    }
}
