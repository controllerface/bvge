package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.gl.Models;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

/**
 * This is the core "factory" class for all physics based objects. It contains named archetype
 * methods that can be used to create tracked objects.
 */
public class PhysicsObjects
{
    private static final Vector2f vector_buffer = new Vector2f();

    public static int FLAG_NONE = 0x00;
    public static int FLAG_STATIC_OBJECT = 0x01;
    public static int FLAG_INTERIOR_EDGE = 0x01;

    public static float distance(float[] a, float[] b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }

    public static int box(float x, float y, float size, int flags)
    {
        // get the box model
        var mdl = Models.get_model_by_index(0);

        // generate the vertices based on the desired size and position
        var p1 = OpenCL.arg_float2(mdl[0] * size + x, mdl[1] * size + y);
        var p2 = OpenCL.arg_float2(mdl[2] * size + x, mdl[3] * size + y);
        var p3 = OpenCL.arg_float2(mdl[4] * size + x, mdl[5] * size + y);
        var p4 = OpenCL.arg_float2(mdl[6] * size + x, mdl[7] * size + y);

        var p1_index = Main.Memory.newPoint(p1);
        var p2_index = Main.Memory.newPoint(p2);
        var p3_index = Main.Memory.newPoint(p3);
        var p4_index = Main.Memory.newPoint(p4);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4);
        var l1 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        // todo: store this reference angle so it can be used to calculate current body rotation
        var angle = MathEX.angleBetween2Lines(l1, l2);


        // box sides
        var start_edge = Main.Memory.newEdge(p1_index, p2_index, distance(p2, p1));
        Main.Memory.newEdge(p2_index, p3_index, distance(p3, p2));
        Main.Memory.newEdge(p3_index, p4_index, distance(p4, p3));
        Main.Memory.newEdge(p4_index, p1_index, distance(p1, p4));

        // corner braces
        Main.Memory.newEdge(p1_index, p3_index, distance(p3, p1), FLAG_INTERIOR_EDGE);
        var end_edge = Main.Memory.newEdge(p2_index, p4_index, distance(p4, p2), FLAG_INTERIOR_EDGE);

        var table = OpenCL.arg_int4(p1_index, p4_index, start_edge, end_edge);
        var transform = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
        var rotation = OpenCL.arg_float2(0, angle);

        int body_id =  Main.Memory.newBody(transform, rotation, table, flags);
        Models.register_model_instance(0, body_id);
        return body_id;
    }

    public static int dynamic_Box(float x, float y, float size)
    {
        return box(x, y, size, FLAG_NONE);
    }

    public static int static_box(float x, float y, float size)
    {
        return box(x, y, size, FLAG_STATIC_OBJECT);
    }

    public static int polygon1(float x, float y, float size)
    {
        // todo: will probably need the model ID stored for the body in order to delete it later
        var mdl = Models.get_model_by_index(1);

        var p1 = OpenCL.arg_float2(mdl[0] * size + x, mdl[1] * size + y);
        var p2 = OpenCL.arg_float2(mdl[2] * size + x, mdl[3] * size + y);
        var p3 = OpenCL.arg_float2(mdl[4] * size + x, mdl[5] * size + y);
        var p4 = OpenCL.arg_float2(mdl[6] * size + x, mdl[7] * size + y);
        var p5 = OpenCL.arg_float2(mdl[8] * size + x, mdl[9] * size + y);

        var p1_index = Main.Memory.newPoint(p1);
        var p2_index = Main.Memory.newPoint(p2);
        var p3_index = Main.Memory.newPoint(p3);
        var p4_index = Main.Memory.newPoint(p4);
        var p5_index = Main.Memory.newPoint(p5);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4, p5);
        var l1 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        // todo: store this reference angle so it can be used to calculate current body rotation
        var angle = MathEX.angleBetween2Lines(l1, l2);
        if (Main.Memory.bodyCount() == 0)
        {
            System.out.println("Initial: " + angle);
        }
        // box sides
        var start_edge = Main.Memory.newEdge(p1_index, p2_index, distance(p2, p1));
        Main.Memory.newEdge(p2_index, p3_index, distance(p3, p2));
        Main.Memory.newEdge(p3_index, p4_index, distance(p4, p3));
        Main.Memory.newEdge(p4_index, p1_index, distance(p1, p4));
        Main.Memory.newEdge(p4_index, p5_index, distance(p4, p5));
        Main.Memory.newEdge(p3_index, p5_index, distance(p3, p5));

        // corner braces
        Main.Memory.newEdge(p1_index, p3_index, distance(p3, p1), FLAG_INTERIOR_EDGE);
        var end_edge = Main.Memory.newEdge(p2_index, p4_index, distance(p4, p2), FLAG_INTERIOR_EDGE);

        var table = OpenCL.arg_int4(p1_index, p5_index, start_edge, end_edge);
        var transform = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
        var rotation = OpenCL.arg_float2(0, angle);

        // todo: register this body index as using the indexed model
        int body_id = Main.Memory.newBody(transform, rotation, table, FLAG_NONE);
        Models.register_model_instance(1, body_id);
        return body_id;
    }
}
