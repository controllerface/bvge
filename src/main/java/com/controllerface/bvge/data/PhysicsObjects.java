package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

/**
 * This is the core "factor" class for all physics based objects. It contains named archetype
 * methods that can be used to create tracked objects.
 */
public class PhysicsObjects
{
    private static final Vector2f vector_buffer = new Vector2f();

    public static int FLAG_NONE = 0x00;
    public static int FLAG_STATIC = 0x01;

    public static float distance(float[] a, float[] b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }

    public static int box(float x, float y, float size, int flags)
    {
        var halfSize = size / 2;

        var p1 = OpenCL.arg_float2(x - halfSize, y - halfSize);
        var p2 = OpenCL.arg_float2(x + halfSize, y - halfSize);
        var p3 = OpenCL.arg_float2(x + halfSize, y + halfSize);
        var p4 = OpenCL.arg_float2(x - halfSize, y + halfSize);

        var p1_index = Main.Memory.newPoint(p1);
        var p2_index = Main.Memory.newPoint(p2);
        var p3_index = Main.Memory.newPoint(p3);
        var p4_index = Main.Memory.newPoint(p4);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4);

        // box sides
        var start_edge = Main.Memory.newEdge(
            p1_index / Main.Memory.Width.POINT,
            p2_index / Main.Memory.Width.POINT,
            distance(p2, p1));

        Main.Memory.newEdge(
            p2_index / Main.Memory.Width.POINT,
            p3_index / Main.Memory.Width.POINT,
            distance(p3, p2));

        Main.Memory.newEdge(
            p3_index / Main.Memory.Width.POINT,
            p4_index / Main.Memory.Width.POINT,
            distance(p4, p3));

        Main.Memory.newEdge(
            p4_index / Main.Memory.Width.POINT,
            p1_index / Main.Memory.Width.POINT,
            distance(p1, p4));

        // corner braces
        Main.Memory.newEdge(
            p1_index / Main.Memory.Width.POINT,
            p3_index / Main.Memory.Width.POINT,
            distance(p3, p1));

        var end_edge = Main.Memory.newEdge(
            p2_index / Main.Memory.Width.POINT,
            p4_index / Main.Memory.Width.POINT,
            distance(p4, p2));

        return Main.Memory.newBody(vector_buffer.x, vector_buffer.y,
            size, size,
            0, 0,
            (float) p1_index / Main.Memory.Width.POINT,
            (float) p4_index / Main.Memory.Width.POINT,
            (float) start_edge / Main.Memory.Width.EDGE,
            (float) end_edge / Main.Memory.Width.EDGE,
            flags);
    }

    public static int dynamic_Box(float x, float y, float size)
    {
        return box(x, y, size, FLAG_NONE);
    }

    public static int static_box(float x, float y, float size)
    {
        return box(x, y, size, FLAG_STATIC);
    }

    public static int polygon1(float x, float y, float size)
    {
        var halfSize = size / 2;

        var p1 = OpenCL.arg_float2(x - halfSize, y - halfSize);
        var p2 = OpenCL.arg_float2(x + halfSize, y - halfSize);
        var p3 = OpenCL.arg_float2(x + halfSize, y + halfSize);
        var p4 = OpenCL.arg_float2(x - halfSize, y + halfSize);
        var p5 = OpenCL.arg_float2(x, y + halfSize * 2);

        var p1_index = Main.Memory.newPoint(p1);
        var p2_index = Main.Memory.newPoint(p2);
        var p3_index = Main.Memory.newPoint(p3);
        var p4_index = Main.Memory.newPoint(p4);
        var p5_index = Main.Memory.newPoint(p5);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4, p5);

        // box sides
        var start_edge = Main.Memory.newEdge(
            p1_index / Main.Memory.Width.POINT,
            p2_index / Main.Memory.Width.POINT,
            distance(p2, p1)
        );

        Main.Memory.newEdge(
            p2_index / Main.Memory.Width.POINT,
            p3_index / Main.Memory.Width.POINT,
            distance(p3, p2)
        );

        Main.Memory.newEdge(
            p3_index / Main.Memory.Width.POINT,
            p4_index / Main.Memory.Width.POINT,
            distance(p4, p3)
        );

        Main.Memory.newEdge(
            p4_index / Main.Memory.Width.POINT,
            p1_index / Main.Memory.Width.POINT,
            distance(p1, p4)
        );


        Main.Memory.newEdge(
            p4_index / Main.Memory.Width.POINT,
            p5_index / Main.Memory.Width.POINT,
            distance(p4, p5)
        );

        Main.Memory.newEdge(
            p3_index / Main.Memory.Width.POINT,
            p5_index / Main.Memory.Width.POINT,
            distance(p3, p5)
        );

        // corner braces
        Main.Memory.newEdge(
            p1_index / Main.Memory.Width.POINT,
            p3_index / Main.Memory.Width.POINT,
            distance(p3, p1)
        );

        var end_edge = Main.Memory.newEdge(
            p2_index / Main.Memory.Width.POINT,
            p4_index / Main.Memory.Width.POINT,
            distance(p4, p2)
        );


        return Main.Memory.newBody(vector_buffer.x, vector_buffer.y,
            size, size,
            0, 0,
            (float) p1_index / Main.Memory.Width.POINT,
            (float) p5_index / Main.Memory.Width.POINT,
            (float) start_edge / Main.Memory.Width.EDGE,
            (float) end_edge / Main.Memory.Width.EDGE,
            FLAG_NONE
        );
    }
}
