package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.geometry.Mesh;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.geometry.Vertex;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.util.MathEX;
import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector4f;

import java.util.*;
import java.util.stream.IntStream;

import static com.controllerface.bvge.geometry.Models.*;

/**
 * This is the core "factory" class for all physics based objects. It contains named archetype
 * methods that can be used to create tracked objects.
 */
public class PhysicsObjects
{
    private static final Vector2f vector_buffer = new Vector2f();

    public static int FLAG_NONE             = Constants.HullFlags.EMPTY.bits;
    public static int FLAG_STATIC_OBJECT    = Constants.HullFlags.IS_STATIC.bits;
    public static int FLAG_CIRCLE           = Constants.HullFlags.IS_CIRCLE.bits;
    public static int FLAG_POLYGON          = Constants.HullFlags.IS_POLYGON.bits;
    public static int FLAG_NO_BONES         = Constants.HullFlags.NO_BONES.bits;


    public static int FLAG_INTERIOR_EDGE = 0x01;

    public static float edgeDistance(float[] a, float[] b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }

    public static int particle(float x, float y, float size, float mass)
    {
        int next_armature_id = Main.Memory.next_armature();
        int next_hull_index = Main.Memory.next_hull();

        // get the circle mesh. this is almost silly to do but just for consistency :-)
        var mesh = Models.get_model_by_index(CIRCLE_PARTICLE).meshes()[0];

        var raw_matrix = CLUtils.arg_float16_matrix(mesh.bone_offsets().get(0).transform());
        int[] bone_index_table = { mesh.bone_offsets().get(0).offset_ref_id(), 0 };
        int bone_id = Main.Memory.new_bone(bone_index_table, raw_matrix);

        var vert = mesh.vertices()[0];

        // the model points are always zero so the * and + are for educational purposes
        var p1 = CLUtils.arg_float2(vert.x() * size + x, vert.y() * size + y);

        var t1 = CLUtils.arg_int2(vert.vert_ref_id(), next_hull_index);

        // store the single point for the circle
        var p1_index = Main.Memory.new_point(p1, t1, new int[4]);

        var edge_index = Main.Memory.new_edge(p1_index, p1_index, edgeDistance(p1, p1));

        var l1 = CLUtils.arg_float4(x, y, x, y + 1);
        var l2 = CLUtils.arg_float4(x, y, p1[0], p1[1]);
        var angle = MathEX.angle_between_lines(l1, l2);
        var table = CLUtils.arg_int4(p1_index, p1_index, edge_index, edge_index);
        var transform = CLUtils.arg_float4(x, y, size, size / 2.0f);
        var rotation = CLUtils.arg_float2(0, angle);

        // there is only one hull, so it is the main hull ID by default
        int[] _flag = CLUtils.arg_int4(FLAG_CIRCLE | FLAG_NO_BONES, next_armature_id, bone_id, bone_id);
        int hull_id = Main.Memory.new_hull(transform, rotation, table, _flag);
        int[] hull_table = CLUtils.arg_int2(hull_id, hull_id);
        int[] armature_flags = CLUtils.arg_int4(hull_id, CIRCLE_PARTICLE, 0, 0);
        return Main.Memory.new_armature(x, y, hull_table, armature_flags, mass);
    }

    public static int tri(float x, float y, float size, int flags, float mass)
    {
        int next_armature_id = Main.Memory.next_armature();
        int next_hull_index = Main.Memory.next_hull();

        // get the box mesh
        var mesh = Models.get_model_by_index(TRIANGLE_PARTICLE).meshes()[0];

        var raw_matrix = CLUtils.arg_float16_matrix(mesh.bone_offsets().get(0).transform());
        int[] bone_index_table = { mesh.bone_offsets().get(0).offset_ref_id(), 0 };
        int bone_id = Main.Memory.new_bone(bone_index_table, raw_matrix);

        var hull = mesh.vertices();
        hull = scale_hull(hull, size);
        hull = translate_hull(hull, x, y);

        var v1 = hull[0];
        var v2 = hull[1];
        var v3 = hull[2];

        var p1 = CLUtils.arg_float2(v1.x(), v1.y());
        var p2 = CLUtils.arg_float2(v2.x(), v2.y());
        var p3 = CLUtils.arg_float2(v3.x(), v3.y());

        var p1_index = Main.Memory.new_point(p1, CLUtils.arg_int2(v1.vert_ref_id(), next_hull_index), new int[4]);
        var p2_index = Main.Memory.new_point(p2, CLUtils.arg_int2(v2.vert_ref_id(), next_hull_index), new int[4]);
        var p3_index = Main.Memory.new_point(p3, CLUtils.arg_int2(v3.vert_ref_id(), next_hull_index), new int[4]);

        MathEX.centroid(vector_buffer, p1, p2, p3);
        var l1 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        var angle = MathEX.angle_between_lines(l1, l2);

        // box sides
        var start_edge = Main.Memory.new_edge(p1_index, p2_index, edgeDistance(p2, p1));
        Main.Memory.new_edge(p2_index, p3_index, edgeDistance(p3, p2));
        var end_edge = Main.Memory.new_edge(p3_index, p1_index, edgeDistance(p3, p1));
//        Main.Memory.new_edge(p3_index, p4_index, edgeDistance(p4, p3));
//        Main.Memory.new_edge(p4_index, p1_index, edgeDistance(p1, p4));

        // corner braces
//        Main.Memory.new_edge(p1_index, p3_index, edgeDistance(p3, p1), FLAG_INTERIOR_EDGE);
//        var end_edge = Main.Memory.new_edge(p2_index, p4_index, edgeDistance(p4, p2), FLAG_INTERIOR_EDGE);

        var table = CLUtils.arg_int4(p1_index, p3_index, start_edge, end_edge);
        var transform = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
        var rotation = CLUtils.arg_float2(0, angle);


        // there is only one hull, so it is the main hull ID by default
        int [] _flag =  CLUtils.arg_int4(flags | FLAG_POLYGON | FLAG_NO_BONES, next_armature_id, bone_id, bone_id);
        int hull_id = Main.Memory.new_hull(transform, rotation, table, _flag);
        int[] hull_table = CLUtils.arg_int2(hull_id, hull_id);
        int[] armature_flags = CLUtils.arg_int4(hull_id, TRIANGLE_PARTICLE, 0,0);
        return Main.Memory.new_armature(x, y, hull_table, armature_flags, mass);
    }

    public static int box(float x, float y, float size, int flags, float mass)
    {
        int next_armature_id = Main.Memory.next_armature();
        int next_hull_index = Main.Memory.next_hull();

        // get the box mesh
        var mesh = Models.get_model_by_index(SQUARE_PARTICLE).meshes()[0];

        var raw_matrix = CLUtils.arg_float16_matrix(mesh.bone_offsets().get(0).transform());
        int[] bone_index_table = { mesh.bone_offsets().get(0).offset_ref_id(), 0 };
        int bone_id = Main.Memory.new_bone(bone_index_table, raw_matrix);

        var hull = calculate_convex_hull(mesh.vertices());
        hull = scale_hull(hull, size);
        hull = translate_hull(hull, x, y);

        var v1 = hull[0];
        var v2 = hull[1];
        var v3 = hull[2];
        var v4 = hull[3];

        var p1 = CLUtils.arg_float2(v1.x(), v1.y());
        var p2 = CLUtils.arg_float2(v2.x(), v2.y());
        var p3 = CLUtils.arg_float2(v3.x(), v3.y());
        var p4 = CLUtils.arg_float2(v4.x(), v4.y());

        var p1_index = Main.Memory.new_point(p1, CLUtils.arg_int2(v1.vert_ref_id(), next_hull_index), new int[4]);
        var p2_index = Main.Memory.new_point(p2, CLUtils.arg_int2(v2.vert_ref_id(), next_hull_index), new int[4]);
        var p3_index = Main.Memory.new_point(p3, CLUtils.arg_int2(v3.vert_ref_id(), next_hull_index), new int[4]);
        var p4_index = Main.Memory.new_point(p4, CLUtils.arg_int2(v4.vert_ref_id(), next_hull_index), new int[4]);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4);
        var l1 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        var angle = MathEX.angle_between_lines(l1, l2);

        // box sides
        var start_edge = Main.Memory.new_edge(p1_index, p2_index, edgeDistance(p2, p1));
        Main.Memory.new_edge(p2_index, p3_index, edgeDistance(p3, p2));
        Main.Memory.new_edge(p3_index, p4_index, edgeDistance(p4, p3));
        Main.Memory.new_edge(p4_index, p1_index, edgeDistance(p1, p4));

        // corner braces
        Main.Memory.new_edge(p1_index, p3_index, edgeDistance(p3, p1), FLAG_INTERIOR_EDGE);
        var end_edge = Main.Memory.new_edge(p2_index, p4_index, edgeDistance(p4, p2), FLAG_INTERIOR_EDGE);

        var table = CLUtils.arg_int4(p1_index, p4_index, start_edge, end_edge);
        var transform = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
        var rotation = CLUtils.arg_float2(0, angle);


        // there is only one hull, so it is the main hull ID by default
        int [] _flag = CLUtils.arg_int4(flags | FLAG_POLYGON, next_armature_id, bone_id, bone_id);
        int hull_id = Main.Memory.new_hull(transform, rotation, table, _flag);
        int[] hull_table = CLUtils.arg_int2(hull_id, hull_id);
        int[] armature_flags = CLUtils.arg_int4(hull_id, SQUARE_PARTICLE, 0, 0);
        return Main.Memory.new_armature(x, y, hull_table, armature_flags, mass);
    }

    public static int dynamic_Box(float x, float y, float size, float mass)
    {
        return box(x, y, size, FLAG_NONE | FLAG_NO_BONES, mass);
    }

    public static int static_box(float x, float y, float size, float mass)
    {
        return box(x, y, size, FLAG_STATIC_OBJECT | FLAG_NO_BONES, mass);
    }

    // todo: add support for boneless models, right now if a model with no bones is loaded, it will
    //  probably break/crash.
    public static int wrap_model(int model_index, float x, float y, float size, int flags, float mass)
    {
        // we need to know the next armature ID before we create it so it can be used for hulls
        // note: like all other memory accessing methods, this relies on single-threaded operation
        int next_armature_id = Main.Memory.next_armature();

        // get the model from the registry
        var model = Models.get_model_by_index(model_index);

        // we need to track which hull is the root hull for this model
        int root_hull_id = -1;
        float root_x = 0;
        float root_y = 0;

        // todo: need some kind of mesh buffer to store their relationships.
        //  will be needed to implement pins/joints.

        var meshes = model.meshes();
        int first_hull = -1;
        int last_hull = -1;
        for (int mesh_index = 0; mesh_index < meshes.length; mesh_index++)
        {
            int next_hull = Main.Memory.next_hull();
            var hull_mesh = meshes[mesh_index];

            // The hull is generated based on the mesh, so it's initial position and rotation
            // are set from the mesh data using its transform. Then, the mesh is scaled to the
            // desired size and moved to the spawn location in world space.
            var hull = generate_convex_hull(hull_mesh);
            hull = transform_hull(hull, hull_mesh.sceneNode().transform);
            hull = scale_hull(hull, size);
            hull = translate_hull(hull, x, y);

            var bone_map = new HashMap<String, Integer>();
            int start_bone = -1;
            int end_bone = -1;
            for (int bone_index = 0; bone_index < hull_mesh.bone_offsets().size(); bone_index++)
            {
                var bone_offset = hull_mesh.bone_offsets().get(bone_index);
                var bone_bind_pose = model.bone_transforms().get(bone_offset.name());
                var bone_transform = bone_bind_pose.mul(bone_offset.transform(), new Matrix4f());
                var raw_matrix = CLUtils.arg_float16_matrix(bone_transform);
                var bind_pose_id = model.bone_indices().get(bone_offset.name());
                int[] bone_table = new int[]{ bone_offset.offset_ref_id(), bind_pose_id };
                int next_bone = Main.Memory.new_bone(bone_table, raw_matrix);
                bone_map.put(bone_offset.name(), next_bone);

                if (start_bone == -1)
                {
                    start_bone = next_bone;
                }
                end_bone = next_bone;
            }

            // generate the points in memory for this object
            int point_start = -1;
            int point_end = -1;
            int[] point_table = new int[hull.length];
            List<float[]> point_buffer = new ArrayList<>();
            for (int point_index = 0; point_index < point_table.length; point_index++)
            {
                var next_vertex = hull[point_index];
                var new_point = CLUtils.arg_float2(next_vertex.x(), next_vertex.y());
                var new_table = CLUtils.arg_int2(next_vertex.vert_ref_id(), next_hull);

                var bone_names = next_vertex.bone_names();
                int[] bone_ids = new int[4];
                bone_ids[0] = bone_names[0] == null ? -1 : bone_map.get(bone_names[0]);
                bone_ids[1] = bone_names[1] == null ? -1 : bone_map.get(bone_names[1]);
                bone_ids[2] = bone_names[2] == null ? -1 : bone_map.get(bone_names[2]);
                bone_ids[3] = bone_names[3] == null ? -1 : bone_map.get(bone_names[3]);
                var next_point = Main.Memory.new_point(new_point, new_table, bone_ids);

                if (point_start == -1)
                {
                    point_start = next_point;
                }
                point_end = next_point;
                point_table[point_index] = next_point;
                point_buffer.add(new_point);
            }

            // generate edges in memory for this object
            int edge_start = -1;
            int edge_end = -1;
            for (int point_1_index = 0; point_1_index < hull.length; point_1_index++)
            {
                int point_2_index = point_1_index + 1;
                if (point_2_index == hull.length)
                {
                    point_2_index = 0;
                }
                var point_1 = point_buffer.get(point_1_index);
                var point_2 = point_buffer.get(point_2_index);
                var distance = edgeDistance(point_2, point_1);
                var next_edge = Main.Memory.new_edge(point_table[point_1_index], point_table[point_2_index], distance);
                if (edge_start == -1)
                {
                    edge_start = next_edge;
                }
                edge_end = next_edge;
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
                    edge_end = Main.Memory.new_edge(point_table[p1_index], point_table[p2_index], distance, FLAG_INTERIOR_EDGE);
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
                edge_end = Main.Memory.new_edge(point_table[p1_index], point_table[p2_index], distance, FLAG_INTERIOR_EDGE);

                if (quarter_count > 1)
                {
                    int p3_index = p1_index + quarter_count;
                    var p3 = point_buffer.get(p3_index);
                    var distance2 = edgeDistance(p3, p1);
                    edge_end = Main.Memory.new_edge(point_table[p1_index], point_table[p3_index], distance2, FLAG_INTERIOR_EDGE);
                }
            }
            if (odd_count) // if there was an odd vertex at the end, connect it to the mid-point
            {
                int p2_index = point_table.length - 1;
                var p1 = point_buffer.get(half_count+1);
                var p2 = point_buffer.get(p2_index);
                var distance = edgeDistance(p2, p1);
                edge_end = Main.Memory.new_edge(point_table[half_count+1], point_table[p2_index], distance, FLAG_INTERIOR_EDGE);
            }

            // calculate centroid and reference angle
            MathEX.centroid(vector_buffer, hull);
            var l1 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
            var l2 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, hull[0].x(), hull[0].y());
            var angle = MathEX.angle_between_lines(l1, l2);

            var table = CLUtils.arg_int4(point_start, point_end, edge_start, edge_end);
            var transform = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, size, size);
            var rotation = CLUtils.arg_float2(0, angle);

            int[] hull_flags = CLUtils.arg_int4(flags, next_armature_id, start_bone, end_bone);
            int hull_id = Main.Memory.new_hull(transform, rotation, table, hull_flags);

            if (first_hull == -1)
            {
                first_hull = hull_id;
            }
            last_hull = hull_id;

            if (next_hull != hull_id)
            {
                throw new RuntimeException("hull/bone alignment error: h=" + hull_id + " b=" + next_hull);
            }
            if (mesh_index == model.root_index())
            {
                root_hull_id = hull_id;
                root_x = vector_buffer.x;
                root_y = vector_buffer.y;
            }
        }

        if (root_hull_id == -1)
        {
            throw new IllegalStateException("There was no root hull determined. "
                + "Check model data to ensure it is correct");
        }

        int[] hull_table = CLUtils.arg_int2(first_hull, last_hull);
        int[] armature_flags = CLUtils.arg_int4(root_hull_id, model_index, 0, 0);
        return Main.Memory.new_armature(root_x, root_y, hull_table, armature_flags, mass);
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
            output[i] = new Vertex(next.vert_ref_id(), vec.x, vec.y, next.uv_data(), next.bone_names(), next.bone_weights());
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

    /**
     * Calculate a convex hull for the provided vertices. The returned vertex array will be a subset
     * of the input array, and may contain all points within the input array, depending on the geometry
     * it describes.
     *
     * @param in_points the points to wrap with in a convex hull
     * @return vertex array that describes the convex hull
     */
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

    /**
     * Calculate a convex hull for the provided vertices, but instead of returning the vertices directly, an
     * index array is returned. This array contains a mapping for each hull vertex to the corresponding vertex
     * in the input vertex array.
     * This is useful for storing the hull data in a more compact format, because the hull is itself made up of
     * vertices that are already present in the meshes data, it is more space efficient to store them as an
     * index array and use this as a lookup table at run time, if the vertex is even needed.
     *
     * @param in_points the points to wrap with in a convex hull
     * @return vertex index table that describes the convex hull
     */
    public static int[] calculate_convex_hull_table(Vertex[] in_points)
    {
        var hull = calculate_convex_hull(in_points);
        var vertex_table = new int[hull.length];
        for (int hull_index = 0; hull_index < hull.length; hull_index++)
        {
            var next_vert = hull[hull_index];
            var next_index = IntStream.range(0, in_points.length)
                .filter(point_index -> in_points[point_index].equals(next_vert))
                .findFirst().orElseThrow(() -> new RuntimeException("Vertex could not be found"));
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
