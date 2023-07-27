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
    private static final Vector2f vectorBuffer = new Vector2f();

    public static int FLAG_NONE = 0x00;
    public static int FLAG_STATIC = 0x01;


    public static float distance(float[] a, float[]b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }


    public static int simpleBox(float x, float y, float size)
    {
        var halfSize = size / 2;

        var p1 = OpenCL.arg_float2(x - halfSize, y - halfSize);
        var p2 = OpenCL.arg_float2(x + halfSize, y - halfSize);
        var p3 = OpenCL.arg_float2(x + halfSize, y + halfSize);
        var p4 = OpenCL.arg_float2(x - halfSize, y + halfSize);

        var p1_obj = Main.Memory.newPoint(p1);
        var p2_obj = Main.Memory.newPoint(p2);
        var p3_obj = Main.Memory.newPoint(p3);
        var p4_obj = Main.Memory.newPoint(p4);

        MathEX.centroid(vectorBuffer, p1, p2, p3, p4);

        // box sides
        var start_edge = Main.Memory.newEdge(
            p1_obj / Main.Memory.Width.POINT,
            p2_obj / Main.Memory.Width.POINT,
            distance(p2, p1)
        );

        Main.Memory.newEdge(
            p2_obj / Main.Memory.Width.POINT,
            p3_obj / Main.Memory.Width.POINT,
            distance(p3, p2)
        );

        Main.Memory.newEdge(
            p3_obj / Main.Memory.Width.POINT,
            p4_obj / Main.Memory.Width.POINT,
            distance(p4, p3)
        );

        Main.Memory.newEdge(
            p4_obj / Main.Memory.Width.POINT,
            p1_obj / Main.Memory.Width.POINT,
            distance(p1, p4)
        );

        // corner braces
        Main.Memory.newEdge(
            p1_obj / Main.Memory.Width.POINT,
            p3_obj / Main.Memory.Width.POINT,
            distance(p3, p1)
        );

        var end_edge = Main.Memory.newEdge(
            p2_obj / Main.Memory.Width.POINT,
            p4_obj / Main.Memory.Width.POINT,
            distance(p4, p2)
        );

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            (float) p1_obj / Main.Memory.Width.POINT,
            (float) p4_obj / Main.Memory.Width.POINT,
            (float) start_edge / Main.Memory.Width.EDGE,
            (float) end_edge / Main.Memory.Width.EDGE,
            FLAG_NONE
        );
    }

    public static int staticBox(float x, float y, float size)
    {
        var halfSize = size / 2;

        var p1 = OpenCL.arg_float2(x - halfSize, y - halfSize);
        var p2 = OpenCL.arg_float2(x + halfSize, y - halfSize);
        var p3 = OpenCL.arg_float2(x + halfSize, y + halfSize);
        var p4 = OpenCL.arg_float2(x - halfSize, y + halfSize);

        var p1_obj = Main.Memory.newPoint(p1);
        var p2_obj = Main.Memory.newPoint(p2);
        var p3_obj = Main.Memory.newPoint(p3);
        var p4_obj = Main.Memory.newPoint(p4);

        MathEX.centroid(vectorBuffer, p1, p2, p3, p4);

        // box sides
        var e1 = Main.Memory.newEdge(
            p1_obj / Main.Memory.Width.POINT,
            p2_obj / Main.Memory.Width.POINT,
            distance(p2, p1)
        );

        var e2 = Main.Memory.newEdge(
            p2_obj / Main.Memory.Width.POINT,
            p3_obj / Main.Memory.Width.POINT,
            distance(p3, p2)
        );

        var e3 = Main.Memory.newEdge(
            p3_obj / Main.Memory.Width.POINT,
            p4_obj / Main.Memory.Width.POINT,
            distance(p4, p3)
        );

        var e4 = Main.Memory.newEdge(
            p4_obj / Main.Memory.Width.POINT,
            p1_obj / Main.Memory.Width.POINT,
            distance(p1, p4)
        );

        // corner braces
        var e5 = Main.Memory.newEdge(
            p1_obj / Main.Memory.Width.POINT,
            p3_obj / Main.Memory.Width.POINT,
            distance(p3, p1)
        );

        var e6 = Main.Memory.newEdge(
            p2_obj / Main.Memory.Width.POINT,
            p4_obj / Main.Memory.Width.POINT,
            distance(p4, p2)
        );

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            (float) p1_obj / Main.Memory.Width.POINT,
            (float) p4_obj / Main.Memory.Width.POINT,
            (float) e1 / Main.Memory.Width.EDGE,
            (float) e6 / Main.Memory.Width.EDGE,
            FLAG_STATIC
        );
    }

    public static int polygon1(float x, float y, float size)
    {
        var halfSize = size / 2;

        var p1 = OpenCL.arg_float2(x - halfSize, y - halfSize);
        var p2 = OpenCL.arg_float2(x + halfSize, y - halfSize);
        var p3 = OpenCL.arg_float2(x + halfSize, y + halfSize);
        var p4 = OpenCL.arg_float2(x - halfSize, y + halfSize);
        var p5 = OpenCL.arg_float2(x, y + halfSize * 2);

        var p1_obj = Main.Memory.newPoint(p1);
        var p2_obj = Main.Memory.newPoint(p2);
        var p3_obj = Main.Memory.newPoint(p3);
        var p4_obj = Main.Memory.newPoint(p4);
        var p5_obj = Main.Memory.newPoint(p5);

        MathEX.centroid(vectorBuffer, p1, p2, p3, p4, p5);

        // box sides
        var e1 = Main.Memory.newEdge(
                p1_obj / Main.Memory.Width.POINT,
                p2_obj / Main.Memory.Width.POINT,
                distance(p2, p1)
        );

        var e2 = Main.Memory.newEdge(
                p2_obj / Main.Memory.Width.POINT,
                p3_obj / Main.Memory.Width.POINT,
                distance(p3, p2)
        );

        var e3 = Main.Memory.newEdge(
                p3_obj / Main.Memory.Width.POINT,
                p4_obj / Main.Memory.Width.POINT,
                distance(p4, p3)
        );

        var e4 = Main.Memory.newEdge(
                p4_obj / Main.Memory.Width.POINT,
                p1_obj / Main.Memory.Width.POINT,
                distance(p1, p4)
        );


        var e5a = Main.Memory.newEdge(
                p4_obj / Main.Memory.Width.POINT,
                p5_obj / Main.Memory.Width.POINT,
                distance(p4, p5)
        );

        var e5b = Main.Memory.newEdge(
                p3_obj / Main.Memory.Width.POINT,
                p5_obj / Main.Memory.Width.POINT,
                distance(p3, p5)
        );

        // corner braces
        var e5 = Main.Memory.newEdge(
                p1_obj / Main.Memory.Width.POINT,
                p3_obj / Main.Memory.Width.POINT,
                distance(p3, p1)
        );

        var e6 = Main.Memory.newEdge(
                p2_obj / Main.Memory.Width.POINT,
                p4_obj / Main.Memory.Width.POINT,
                distance(p4, p2)
        );


        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            (float) p1_obj / Main.Memory.Width.POINT,
            (float) p5_obj / Main.Memory.Width.POINT,
            (float) e1 / Main.Memory.Width.EDGE,
            (float) e6 / Main.Memory.Width.EDGE,
            FLAG_NONE
        );
    }
}
