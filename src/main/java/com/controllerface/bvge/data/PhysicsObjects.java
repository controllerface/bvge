package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.geometry.Meshes;
import com.controllerface.bvge.geometry.Models;
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
    public static int FLAG_CIRCLE = 0x02;
    public static int FLAG_POLYGON = 0x04;
    public static int FLAG_INTERIOR_EDGE = 0x01;

    public static float edgeDistance(float[] a, float[] b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }

    public static int circle(float x, float y, float size)
    {
        // get the circle model. this is almost silly to do but just for consistency :-)
        var mesh = Models.get_model_by_index(Models.CIRCLE_MODEL).meshes()[0];

        // the model points are always zero so the * and + are for educational purposes
        var p1 = OpenCL.arg_float2(mesh.vertices()[0].x() * size + x, mesh.vertices()[0].y() * size + y);

        // store the single point for the circle
        var p1_index = Main.Memory.newPoint(p1);
        var l1 = OpenCL.arg_float4(x, y, x, y + 1);
        var l2 = OpenCL.arg_float4(x, y, p1[0], p1[1]);
        var angle = MathEX.angleBetween2Lines(l1, l2);
        var table = OpenCL.arg_int4(p1_index, p1_index, -1, -1);
        var transform = OpenCL.arg_float4(x, y, size, size / 2.0f);
        var rotation = OpenCL.arg_float2(0, angle);

        // todo: future uses of this class should be decoupled from the model registry itself
        //  objects will need to be separated into visual and physics components, with the current
        //  "body" objects being used for the physics bounds, i.e. convex hulls
        int hull_id = Main.Memory.newHull(transform, rotation, table, FLAG_CIRCLE);
        Models.register_model_instance(Models.CIRCLE_MODEL, hull_id);
        return hull_id;
    }

    public static int box(float x, float y, float size, int flags)
    {
        // get the box model
        var mesh = Models.get_model_by_index(Models.BOX_MODEL).meshes()[0];

        // generate the vertices based on the desired size and position
        var p1 = OpenCL.arg_float2(mesh.vertices()[0].x() * size + x, mesh.vertices()[0].y() * size + y);
        var p2 = OpenCL.arg_float2(mesh.vertices()[1].x() * size + x, mesh.vertices()[1].y() * size + y);
        var p3 = OpenCL.arg_float2(mesh.vertices()[2].x() * size + x, mesh.vertices()[2].y() * size + y);
        var p4 = OpenCL.arg_float2(mesh.vertices()[3].x() * size + x, mesh.vertices()[3].y() * size + y);

        var p1_index = Main.Memory.newPoint(p1);
        var p2_index = Main.Memory.newPoint(p2);
        var p3_index = Main.Memory.newPoint(p3);
        var p4_index = Main.Memory.newPoint(p4);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4);
        var l1 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        var angle = MathEX.angleBetween2Lines(l1, l2);

        // box sides
        var start_edge = Main.Memory.newEdge(p1_index, p2_index, edgeDistance(p2, p1));
        Main.Memory.newEdge(p2_index, p3_index, edgeDistance(p3, p2));
        Main.Memory.newEdge(p3_index, p4_index, edgeDistance(p4, p3));
        Main.Memory.newEdge(p4_index, p1_index, edgeDistance(p1, p4));

        // corner braces
        Main.Memory.newEdge(p1_index, p3_index, edgeDistance(p3, p1), FLAG_INTERIOR_EDGE);
        var end_edge = Main.Memory.newEdge(p2_index, p4_index, edgeDistance(p4, p2), FLAG_INTERIOR_EDGE);

        var table = OpenCL.arg_int4(p1_index, p4_index, start_edge, end_edge);
        var transform = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
        var rotation = OpenCL.arg_float2(0, angle);

        // todo: register a new model instead of a new mesh, move over to renderers loading models
        //  instead of meshes
        int hull_id =  Main.Memory.newHull(transform, rotation, table, flags | FLAG_POLYGON);
        Models.register_model_instance(Models.BOX_MODEL, hull_id);
        return hull_id;
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
        var mesh = Models.get_model_by_index(Models.POLYGON1_MODEL).meshes()[0];

        var p1 = OpenCL.arg_float2(mesh.vertices()[0].x() * size + x, mesh.vertices()[0].x() * size + y);
        var p2 = OpenCL.arg_float2(mesh.vertices()[1].x() * size + x, mesh.vertices()[1].y() * size + y);
        var p3 = OpenCL.arg_float2(mesh.vertices()[2].x() * size + x, mesh.vertices()[2].y() * size + y);
        var p4 = OpenCL.arg_float2(mesh.vertices()[3].x() * size + x, mesh.vertices()[3].y() * size + y);
        var p5 = OpenCL.arg_float2(mesh.vertices()[4].x() * size + x, mesh.vertices()[4].y() * size + y);

        var p1_index = Main.Memory.newPoint(p1);
        var p2_index = Main.Memory.newPoint(p2);
        var p3_index = Main.Memory.newPoint(p3);
        var p4_index = Main.Memory.newPoint(p4);
        var p5_index = Main.Memory.newPoint(p5);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4, p5);
        var l1 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        var angle = MathEX.angleBetween2Lines(l1, l2);
        // box sides
        var start_edge = Main.Memory.newEdge(p1_index, p2_index, edgeDistance(p2, p1));
        Main.Memory.newEdge(p2_index, p3_index, edgeDistance(p3, p2));
        Main.Memory.newEdge(p3_index, p4_index, edgeDistance(p4, p3));
        Main.Memory.newEdge(p4_index, p1_index, edgeDistance(p1, p4));
        Main.Memory.newEdge(p4_index, p5_index, edgeDistance(p4, p5));
        Main.Memory.newEdge(p3_index, p5_index, edgeDistance(p3, p5));

        // corner braces
        Main.Memory.newEdge(p1_index, p3_index, edgeDistance(p3, p1), FLAG_INTERIOR_EDGE);
        var end_edge = Main.Memory.newEdge(p2_index, p4_index, edgeDistance(p4, p2), FLAG_INTERIOR_EDGE);

        var table = OpenCL.arg_int4(p1_index, p5_index, start_edge, end_edge);
        var transform = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
        var rotation = OpenCL.arg_float2(0, angle);

        // todo: register a new model instead of a new mesh, move over to renderers loading models
        //  instead of meshes
        int hull_id = Main.Memory.newHull(transform, rotation, table, FLAG_NONE | FLAG_POLYGON);
        Models.register_model_instance(Models.POLYGON1_MODEL, hull_id);
        return hull_id;
    }
}
