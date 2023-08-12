package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.OpenCL;
import com.controllerface.bvge.geometry.Mesh;
import com.controllerface.bvge.geometry.Model;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.geometry.Vertex;
import com.controllerface.bvge.util.MathEX;
import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector4f;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static com.controllerface.bvge.geometry.Models.CRATE_MODEL;

/**
 * This is the core "factory" class for all physics based objects. It contains named archetype
 * methods that can be used to create tracked objects.
 */
public class PhysicsObjects
{
    private static final Vector2f vector_buffer = new Vector2f();

    // todo: this will need some kind of protection from overflow, or re-use of IDs.
    //  it will likely be the case that objects will eventually spawn and get despawned,
    //  but IDs like this will not be re-used so eventually the game could run out of ids.
    //  ideally, this would be set to some ID that actually is associated directly with
    //  the model but the current design makes this hard to do.

    private static AtomicInteger next_model_id = new AtomicInteger(0);

    public static int FLAG_NONE = 0x00;
    public static int FLAG_STATIC_OBJECT = 0x01;
    public static int FLAG_CIRCLE = 0x02;
    public static int FLAG_POLYGON = 0x04;
    public static int FLAG_INTERIOR_EDGE = 0x01;

    public static float edgeDistance(float[] a, float[] b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }

    public static int particle(float x, float y, float size)
    {
        // get the circle mesh. this is almost silly to do but just for consistency :-)
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

        // there is only one hull, so it is the main hull ID by default
        int[] _flag = OpenCL.arg_int2(FLAG_CIRCLE, next_model_id.getAndIncrement());
        int hull_id = Main.Memory.newHull(transform, rotation, table, _flag);
        Models.register_model_instance(Models.CIRCLE_MODEL, hull_id);
        return hull_id;
    }

    public static int box(float x, float y, float size, int flags)
    {
        // get the box mesh
        var mesh = Models.get_model_by_index(CRATE_MODEL).meshes()[0];
        var hull = calculate_convex_hull(mesh.vertices());
        hull = scale_hull(hull, size);
        hull = translate_hull(hull, x, y);

        var p1 = OpenCL.arg_float2(hull[0].x(), hull[0].y());
        var p2 = OpenCL.arg_float2(hull[1].x(), hull[1].y());
        var p3 = OpenCL.arg_float2(hull[2].x(), hull[2].y());
        var p4 = OpenCL.arg_float2(hull[3].x(), hull[3].y());

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

        // there is only one hull, so it is the main hull ID by default
        int [] _flag =  OpenCL.arg_int2(flags | FLAG_POLYGON, next_model_id.getAndIncrement());
        int hull_id = Main.Memory.newHull(transform, rotation, table, _flag);
        Models.register_model_instance(CRATE_MODEL, hull_id);
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

    public static int wrap_model(int model_index, float x, float y, float size, int flags)
    {
        int model_instance_id = next_model_id.getAndIncrement();

        // get the model from the registry
        var model = Models.get_model_by_index(model_index);

        // we need to track which hull is the root hull for this model
        int root_hull_id = -1;

        // loop through each mesh and generate a hull for it
        var meshes = model.meshes();
        for (int i = 0; i < meshes.length; i++)
        {
            // get the next mesh
            var next_mesh = meshes[i];

            // generate the hull
            var hull = generate_convex_hull(next_mesh);

            // This alternative to using the bone transform is using the mesh transform
            // directly. This is helpful for debugging as the initial bone transform
            // outcome on the mesh should be identical this reference position.
            //
            //hull = transform_hull(hull, next_mesh.sceneNode().transform);

            // todo: create Bone memory structure
            // use bone transform to position the hull
            var bone_transform = model.bone_transforms().get(next_mesh.bone().name());
            hull = transform_hull(hull, bone_transform);

            // todo: determine how to re-apply bone transform after scaling

            // todo: adapt tutorials usage of bone matrix to compute positions in CL
            //  assume all bones are combined into one offset and avoid needing the
            //  index.

            // scale to desired size in model space
            hull = scale_hull(hull, size);

            // translate to world space
            hull = translate_hull(hull, x, y);

            // generate the points in memory for this object
            int start_point = -1;
            int end_point = -1;
            int[] point_table = new int[hull.length];
            List<float[]> point_buffer = new ArrayList<>();
            for (int point_index = 0; point_index < point_table.length; point_index++)
            {
                var next_vertex = hull[point_index];
                var new_point = OpenCL.arg_float2(next_vertex.x(), next_vertex.y());
                var p_index = Main.Memory.newPoint(new_point);
                if (start_point == -1)
                {
                    start_point = p_index;
                }
                end_point = p_index;
                point_table[point_index] = p_index;
                point_buffer.add(new_point);
            }

            // generate edges in memory for this object
            int start_edge = -1;
            int end_edge = -1;
            for (int edge_index = 0; edge_index < hull.length; edge_index++)
            {
                int p1_index = edge_index;
                int p2_index = edge_index + 1;
                if (p2_index == hull.length)
                {
                    p2_index = 0;
                }
                var p1 = point_buffer.get(p1_index);
                var p2 = point_buffer.get(p2_index);
                var distance = edgeDistance(p2, p1);
                var e_index = Main.Memory.newEdge(point_table[p1_index], point_table[p2_index], distance);
                if (start_edge == -1)
                {
                    start_edge = e_index;
                }
                end_edge = e_index;
            }

            // calculate interior edges

            // connect every other
            if (hull.length > 4)
            {
                //pass 1
                for (int p1_index = 0; p1_index < hull.length; p1_index++)
                {
                    int p2_index = p1_index + 2;
                    if (p2_index > point_buffer.size() - 1)
                    {
                        continue;
                    }
                    var p1 = point_buffer.get(p1_index);
                    var p2 = point_buffer.get(p2_index);
                    var distance = edgeDistance(p2, p1);
                    end_edge = Main.Memory.newEdge(point_table[p1_index], point_table[p2_index], distance, FLAG_INTERIOR_EDGE);
                }
            }

            // pass 2
            boolean odd_count = hull.length % 2 != 0;
            int half_count = hull.length / 2;
            int quarter_count = half_count / 2;
            for (int p1_index = 0; p1_index < half_count; p1_index++)
            {
                int p2_index = p1_index + half_count;
                var p1 = point_buffer.get(p1_index);
                var p2 = point_buffer.get(p2_index);
                var distance = edgeDistance(p2, p1);
                end_edge = Main.Memory.newEdge(point_table[p1_index], point_table[p2_index], distance, FLAG_INTERIOR_EDGE);

                if (quarter_count > 1)
                {
                    int p3_index = p1_index + quarter_count;
                    var p3 = point_buffer.get(p3_index);
                    var distance2 = edgeDistance(p3, p1);
                    end_edge = Main.Memory.newEdge(point_table[p1_index], point_table[p3_index], distance2, FLAG_INTERIOR_EDGE);
                }
            }
            if (odd_count) // if there was an odd vertex at the end, connect it to the mid point
            {
                int p2_index = point_table.length - 1;
                var p1 = point_buffer.get(half_count+1);
                var p2 = point_buffer.get(p2_index);
                var distance = edgeDistance(p2, p1);
                end_edge = Main.Memory.newEdge(point_table[half_count+1], point_table[p2_index], distance, FLAG_INTERIOR_EDGE);
            }

            // calculate centroid and reference angle
            MathEX.centroid(vector_buffer, hull);
            var l1 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
            var l2 = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, hull[0].x(), hull[0].y());
            var angle = MathEX.angleBetween2Lines(l1, l2);

            var table = OpenCL.arg_int4(start_point, end_point, start_edge, end_edge);
            var transform = OpenCL.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
            var rotation = OpenCL.arg_float2(0, angle);

            int[] _flag = OpenCL.arg_int2(flags, model_instance_id);
            // there is only one hull, so it is the main hull ID by default
            int hull_id = Main.Memory.newHull(transform, rotation, table, _flag);
            if (i == model.root_index())
            {
                root_hull_id = hull_id;
            }
        }

        if (root_hull_id == -1)
        {
            throw new IllegalStateException("There was no root hull determined. "
                + "Check model data to ensure it is correct");
        }

        Models.register_model_instance(model_index, root_hull_id);
        return root_hull_id;
    }


    public static Vertex[] scale_hull(Vertex[] input, float scale)
    {
        var output = new Vertex[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = input[i].uniform_scale(scale);
        }
        return output;
    }

    public static Vertex[] translate_hull(Vertex[] input, float tx, float ty)
    {
        var output = new Vertex[input.length];
        for (int i = 0; i < input.length; i++)
        {
            output[i] = input[i].translate(tx, ty);
        }
        return output;
    }

    public static Vertex[] transform_hull(Vertex[] input, Matrix4f matrix4f)
    {
        var output = new Vertex[input.length];
        for (int i = 0; i < input.length; i++)
        {
            var next = input[i];
            var vec = matrix4f.transform(new Vector4f(next.x(), next.y(), 0.0f, 1.0f));
            output[i] = new Vertex(vec.x, vec.y, next.bone_name(), next.bone_weight());
        }
        return output;
    }

    // below was pasted and modified from here:
    // https://github.com/rolandopalermo/convex-hull-algorithms/blob/master/src/main/java/com/rolandopalermo/algorithms/convexhull/graphics2D/GiftWrapping.java

    public static int lowestPoint(Vertex[] points)
    {
        Vertex lowest = points[0];
        int index = 0;
        for (int i = 1; i < points.length; i++)
        {
            if (points[i].y() < lowest.y() || (points[i].y() == lowest.y() && points[i].x() > lowest.x()))
            {
                lowest = points[i];
                index = i;
            }
        }
        return index;
    }

    public static Vertex[] swap(Vertex[] points, int index0, int index1)
    {
        Vertex temp = points[index0];
        points[index0] = points[index1];
        points[index1] = temp;
        return points;
    }

    public static Vertex[] generate_convex_hull(Mesh mesh)
    {
        var out = new Vertex[mesh.hull().length];
        for (int i = 0; i < mesh.hull().length; i++)
        {
            var next_index = mesh.hull()[i];
            out[i] = mesh.vertices()[next_index];
        }
        return out;
    }

    public static Vertex[] calculate_convex_hull(Vertex[] in_points)
    {
        // working objects for the loop.
        // p is the current vertex of the calculated hull
        // q is the next vertex of the calculated hull
        Vertex p;
        Vertex q;

        // during hull creation, this holds the vertices that are currently designated as the hull
        Stack<Vertex> hull_vertices = new Stack<>();

        // because the input array is not intended to be changed, we make a copy of the input values
        // and operate on the copy. This is needed because of the swap() calls, which will re-order
        // the vertices in-place to aid with hull calculation.
        Vertex[] points = new Vertex[in_points.length];
        System.arraycopy(in_points, 0, points, 0, in_points.length);

        // do the initial swap to set up the points array for processing,
        // this is essentially one iteration of the loop
        swap(points, 0, lowestPoint(points));
        swap(points, 1, lowestPolarAngle(points, points[0]));

        // init the working data for the loop, pushing the first vertex into the result buffer
        int index = 0;
        p = points[0];
        q = points[1];
        hull_vertices.push(p);

        // loop until the calculated hull makes a loop around the mesh
        while (!points[0].equals(q))
        {
            // push the next vertex into the buffer, since it has been calculated
            hull_vertices.push(q);

            // now iterate through the points and find the next candidate
            double minorPolarAngle = 180D;
            for (int i = points.length - 1; i >=0; i--)
            {
                if (!points[i].equals(q))
                {
                    double angle = 180D - q.angle_between(p, points[i]);
                    if (angle < minorPolarAngle)
                    {
                        minorPolarAngle = angle;
                        index = i;
                    }
                }
            }

            // once a candidate has been found, swap out old current and swap in new next
            p = q;
            q = points[index];
        }

        return hull_vertices.toArray(Vertex[]::new);
    }

    public static int[] calculate_convex_hull_table(Vertex[] in_points)
    {
        var hull = calculate_convex_hull(in_points);
        var vertex_table = new int[hull.length];
        for (int hull_index = 0; hull_index < hull.length; hull_index++)
        {
            var next_vert = hull[hull_index];
            var next_index = IntStream.range(0, in_points.length)
                .filter(point_index -> in_points[point_index].equals(next_vert))
                .findFirst().orElseThrow();
            vertex_table[hull_index] = next_index;
        }
        return vertex_table;
    }

    private static int lowestPolarAngle(Vertex[] points, Vertex lowestPoint)
    {
        int index = 0;
        double minorPolarAngle = 180D;
        for (int i = 1; i < points.length; i++)
        {
            double angle = lowestPoint.angle_between(points[i]);
            if (angle < minorPolarAngle)
            {
                minorPolarAngle = angle;
                index = i;
            }
        }
        return index;
    }
}
