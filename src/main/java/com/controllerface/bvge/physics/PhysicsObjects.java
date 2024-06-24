package com.controllerface.bvge.physics;

import com.controllerface.bvge.animation.BoneBindPose;
import com.controllerface.bvge.cl.CLUtils;
import com.controllerface.bvge.game.world.sectors.SectorContainer;
import com.controllerface.bvge.animation.AnimationState;
import com.controllerface.bvge.geometry.Mesh;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.geometry.UnloadedEntity;
import com.controllerface.bvge.geometry.Vertex;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.substances.SubstanceTools;
import com.controllerface.bvge.util.MathEX;
import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector4f;

import java.util.*;
import java.util.stream.IntStream;

import static com.controllerface.bvge.geometry.ModelRegistry.*;

import static com.controllerface.bvge.util.Constants.*;

/**
 * This is the core "factory" class for all physics based objects. It contains named archetype
 * methods that can be used to create tracked objects.
 */
public class PhysicsObjects
{
    private static final Vector2f vector_buffer = new Vector2f();
    private static final Matrix4f matrix_buffer = new Matrix4f();
    private static final List<float[]> convex_buffer = new ArrayList<>();
    private static final Stack<Vertex> hull_vertex_buffer = new Stack<>();

    private static final int[] EMPTY_POINT_BONE_TABLE = new int[]{ -1, -1, -1, -1 };

    public static float edgeDistance(float[] a, float[] b)
    {
        return Vector2f.distance(a[0], a[1], b[0], b[1]);
    }

    public static int particle(SectorContainer world,
                               float x,
                               float y,
                               float size,
                               float mass,
                               float friction,
                               float restitution,
                               int range_link,
                               int entity_flags,
                               int global_hull_flags,
                               int point_flags,
                               int model_id,
                               int uv_variant,
                               int type)
    {
        int next_entity_id = world.next_entity();
        int next_hull_index = world.next_hull();

        // get the circle mesh. this is almost silly to do but just for consistency :-)
        var mesh = ModelRegistry.get_model_by_index(model_id).meshes()[0];

        var vert = mesh.vertices()[0];

        // the model points are always zero so the * and + are for educational purposes
        var p1 = CLUtils.arg_float2(vert.x() * size + x, vert.y() * size + y);

        // store the single point for the circle
        var p1_index = world.create_point(p1, EMPTY_POINT_BONE_TABLE, vert.index(), next_hull_index, 0, point_flags);

        var l1 = CLUtils.arg_float4(x, y, x, y + 1);
        var l2 = CLUtils.arg_float4(x, y, p1[0], p1[1]);
        var angle = MathEX.angle_between_lines(l1, l2);
        var point_table = CLUtils.arg_int2(p1_index, p1_index);
        var edge_table = CLUtils.arg_int2(0, -1);
        var position = CLUtils.arg_float2(x, y);
        var scale = CLUtils.arg_float2(size, size / 2.0f);
        var rotation = CLUtils.arg_float2(0, angle);

        // there is only one hull, so it is the main hull ID by default
        int[] bone_table = CLUtils.arg_int2(0, -1);
        int hull_flags = global_hull_flags | HullFlags.IS_CIRCLE._int | HullFlags.NO_BONES._int;
        int hull_id = world.create_hull(mesh.mesh_id(),
            position,
            scale,
            rotation,
            point_table,
            edge_table,
            bone_table,
            friction,
            restitution,
            next_entity_id,
            uv_variant,
            hull_flags);
        int[] hull_table = CLUtils.arg_int2(hull_id, hull_id);

        return world.create_entity(x, y, x, y,
            hull_table,
            bone_table,
            mass,
            -1,
            -1f,
            hull_id,
            model_id,
            range_link,
            type,
            entity_flags);
    }

    public static int liquid_particle(SectorContainer world,
                                      float x,
                                      float y,
                                      float size,
                                      float mass,
                                      float friction,
                                      float restitution,
                                      int entity_flags,
                                      int hull_flags,
                                      int point_flags,
                                      Liquid particle_fluid)
    {
        int type = SubstanceTools.to_type_index(particle_fluid);
        return particle(world, x, y, size, mass, friction, restitution, 0, entity_flags, hull_flags, point_flags, CIRCLE_PARTICLE, particle_fluid.liquid_number, type);
    }

    public static int circle_cursor(SectorContainer world,
                                    float x,
                                    float y,
                                    float size,
                                    int range_link)
    {
        return particle(world, x, y, size, 0.0f, 0.0f, 0.0f, range_link, 0,  HullFlags.IS_CURSOR._int, 0, CURSOR, 0,-1);
    }

    public static int tri(SectorContainer world,
                          float x,
                          float y,
                          float size,
                          int entity_flags,
                          int global_hull_flags,
                          float mass,
                          float friction,
                          float restitution,
                          int model_id,
                          Solid shard_mineral)
    {
        int type = SubstanceTools.to_type_index(shard_mineral);
        int next_entity_id = world.next_entity();
        int next_hull_index = world.next_hull();

        var mesh = ModelRegistry.get_model_by_index(model_id).meshes()[0];

        var hull = mesh.vertices();
        hull = scale_hull(hull, size);
        hull = translate_hull(hull, x, y);

        var v1 = hull[0];
        var v2 = hull[1];
        var v3 = hull[2];

        var p1 = CLUtils.arg_float2(v1.x(), v1.y());
        var p2 = CLUtils.arg_float2(v2.x(), v2.y());
        var p3 = CLUtils.arg_float2(v3.x(), v3.y());

        int h1 = 0;
        int h2 = 0;
        int h3 = 0;

        var p1_index = world.create_point(p1, EMPTY_POINT_BONE_TABLE, v1.index(), next_hull_index, h1,0);
        var p2_index = world.create_point(p2, EMPTY_POINT_BONE_TABLE, v2.index(), next_hull_index, h2,0);
        var p3_index = world.create_point(p3, EMPTY_POINT_BONE_TABLE, v3.index(), next_hull_index, h3,0);

        MathEX.centroid(vector_buffer, p1, p2, p3);
        var l1 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        var angle = MathEX.angle_between_lines(l1, l2);

        var start_edge = world.create_edge(p1_index, p2_index, edgeDistance(p2, p1), 0);
        world.create_edge(p2_index, p3_index, edgeDistance(p3, p2), 0);
        var end_edge = world.create_edge(p3_index, p1_index, edgeDistance(p3, p1), 0);

        var point_table = CLUtils.arg_int2(p1_index, p3_index);
        var edge_table = CLUtils.arg_int2(start_edge, end_edge);
        var position = CLUtils.arg_float2(vector_buffer.x, vector_buffer.y);
        var scale = CLUtils.arg_float2(size, size);
        var rotation = CLUtils.arg_float2(0, angle);

        // there is only one hull, so it is the main hull ID by default
        int[] bone_table = CLUtils.arg_int2(0, -1);
        int hull_flags = global_hull_flags | HullFlags.IS_POLYGON._int | HullFlags.NO_BONES._int;
        int hull_id = world.create_hull(mesh.mesh_id(),
            position,
            scale,
            rotation,
            point_table,
            edge_table,
            bone_table,
            friction,
            restitution,
            next_entity_id,
            shard_mineral.mineral_number,
            hull_flags);
        int[] hull_table = CLUtils.arg_int2(hull_id, hull_id);
        return world.create_entity(x, y, x, y,
            hull_table,
            bone_table,
            mass,
            -1,
            -1f,
            hull_id,
            model_id,
            0,
            type,
            entity_flags);
    }

    public static int block(SectorContainer world,
                            float x,
                            float y,
                            float size,
                            int entity_flags,
                            int global_hull_flags,
                            float mass,
                            float friction,
                            float restitution,
                            int model_id, Solid block_mineral,
                            int[] hits)
    {
        int type = SubstanceTools.to_type_index(block_mineral);
        int next_entity_id = world.next_entity();
        int next_hull_index = world.next_hull();

        // get the box mesh
        var mesh = ModelRegistry.get_model_by_index(model_id).meshes()[0];

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

        var p1_index = world.create_point(p1, EMPTY_POINT_BONE_TABLE, v1.index(), next_hull_index, hits[0],0);
        var p2_index = world.create_point(p2, EMPTY_POINT_BONE_TABLE, v2.index(), next_hull_index, hits[1],0);
        var p3_index = world.create_point(p3, EMPTY_POINT_BONE_TABLE, v3.index(), next_hull_index, hits[2],0);
        var p4_index = world.create_point(p4, EMPTY_POINT_BONE_TABLE, v4.index(), next_hull_index, hits[3],0);

        MathEX.centroid(vector_buffer, p1, p2, p3, p4);
        var l1 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
        var l2 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, p1[0], p1[1]);

        var angle = MathEX.angle_between_lines(l1, l2);

        // box sides
        var start_edge = world.create_edge(p1_index, p2_index, edgeDistance(p2, p1), 0);
        world.create_edge(p2_index, p3_index, edgeDistance(p3, p2), 0);
        world.create_edge(p3_index, p4_index, edgeDistance(p4, p3), 0);
        world.create_edge(p4_index, p1_index, edgeDistance(p1, p4), 0);

        // corner braces
        world.create_edge(p1_index, p3_index, edgeDistance(p3, p1), EdgeFlags.IS_INTERIOR.bits);
        var end_edge = world.create_edge(p2_index, p4_index, edgeDistance(p4, p2), EdgeFlags.IS_INTERIOR.bits);

        var point_table = CLUtils.arg_int2(p1_index, p4_index);
        var edge_table = CLUtils.arg_int2(start_edge, end_edge);
        var position = CLUtils.arg_float2(vector_buffer.x, vector_buffer.y);
        var scale = CLUtils.arg_float2(size, size);
        var rotation = CLUtils.arg_float2(0, angle);

        // there is only one hull, so it is the main hull ID by default
        int[] bone_table = CLUtils.arg_int2(0, -1);
        int hull_flags = global_hull_flags | HullFlags.IS_POLYGON._int;
        int hull_id = world.create_hull(mesh.mesh_id(),
            position,
            scale,
            rotation,
            point_table,
            edge_table,
            bone_table,
            friction,
            restitution,
            next_entity_id,
            block_mineral.mineral_number,
            hull_flags);
        int[] hull_table = CLUtils.arg_int2(hull_id, hull_id);
        return world.create_entity(x, y, x, y,
            hull_table,
            bone_table,
            mass,
            -1,
            -1f,
            hull_id,
            model_id,
            0,
            type,
            entity_flags);
    }


    public static int base_block(SectorContainer world, float x, float y, float size, float mass, float friction, float restitution, int entity_flags, int hull_flags, Solid block_material, int[] hits)
    {
        return block(world, x, y, size, entity_flags,hull_flags | HullFlags.IS_BLOCK._int | HullFlags.NO_BONES._int, mass, friction, restitution, BASE_BLOCK_INDEX, block_material, hits);
    }

    private static final Random random = new Random();

    // todo: add support for boneless models, right now if a model with no bones is loaded, it will
    //  probably break/crash.
    public static int[] wrap_model(SectorContainer world, int model_index, float x, float y, float size, float mass, float friction, float restitution, int uv_offset, int flags)
    {
        // we need to know the next entity ID before we create it, so it can be used for hulls
        // note: like all other memory accessing methods, this relies on single-threaded operation
        int next_entity_id = world.next_entity();

        // get the model from the registry
        var model = ModelRegistry.get_model_by_index(model_index);

        // we need to track which hull is the root hull for this model
        int root_hull_id = -1;

        var meshes = model.meshes();
        int first_hull = -1;
        int last_hull = -1;
        int first_armature_bone = -1;
        int last_armature_bone = -1;

        var armature_bone_map = new HashMap<String, Integer>();
        var armature_bone_parent_map = new HashMap<Integer, Integer>();
        for (Map.Entry<Integer, BoneBindPose> entry : model.bind_poses().entrySet())
        {
            var bind_pose_ref_id = entry.getKey();
            var bind_pose = entry.getValue();
            var raw_matrix = CLUtils.arg_float16_matrix(bind_pose.transform());
            int bind_parent = bind_pose.parent() == -1
                ? -1
                : armature_bone_parent_map.get(bind_pose.parent());

            int next_armature_bone = world.create_entity_bone(bind_pose_ref_id, bind_parent, raw_matrix);
            if (first_armature_bone == -1)
            {
                first_armature_bone = next_armature_bone;
            }
            last_armature_bone = next_armature_bone;

            armature_bone_map.put(bind_pose.bone_name(), next_armature_bone);
            armature_bone_parent_map.put(bind_pose_ref_id, next_armature_bone);
        }

        for (int mesh_index = 0; mesh_index < meshes.length; mesh_index++)
        {
            int local_hull_flags = 0;
            int next_hull = world.next_hull();
            var hull_mesh = meshes[mesh_index];

            if (hull_mesh.name().toLowerCase().contains("hand"))
            {
                local_hull_flags |= HullFlags.IS_HAND._int;
            }

            if (hull_mesh.name().toLowerCase().contains("foot"))
            {
                local_hull_flags |= HullFlags.IS_FOOT._int;
            }

            if (hull_mesh.name().toLowerCase().contains(".r"))
            {
                local_hull_flags |= HullFlags.SIDE_R._int;
            }

            if (hull_mesh.name().toLowerCase().contains(".l"))
            {
                local_hull_flags |= HullFlags.SIDE_L._int;
            }

            // The hull is generated based on the mesh, so it's initial position and rotation
            // are set from the mesh data using its transform. Then, the mesh is scaled to the
            // desired size and moved to the spawn location in world space.
            var new_mesh = transform_hull(hull_mesh.vertices(), hull_mesh.sceneNode().transform);
            new_mesh = scale_hull(new_mesh, size);
            new_mesh = translate_hull(new_mesh, x, y);
            var new_hull = generate_convex_hull(hull_mesh, new_mesh);
            var new_interior_hull = generate_interior_hull(hull_mesh, new_mesh);

            var bone_map = new HashMap<String, Integer>();
            int start_hull_bone = -1;
            int end_hull_bone = -1;
            for (int bone_index = 0; bone_index < hull_mesh.bone_offsets().size(); bone_index++)
            {
                var bone_offset = hull_mesh.bone_offsets().get(bone_index);
                var bone_bind_pose = model.bone_transforms().get(bone_offset.name());
                var bone_transform = bone_bind_pose.mul(bone_offset.transform(), matrix_buffer);
                var raw_matrix = CLUtils.arg_float16_matrix(bone_transform);
                var bind_pose_id = armature_bone_map.get(bone_offset.name());
                int next_bone = world.create_hull_bone(raw_matrix, bind_pose_id, bone_offset.offset_ref_id());
                bone_map.put(bone_offset.name(), next_bone);

                if (start_hull_bone == -1)
                {
                    start_hull_bone = next_bone;
                }
                end_hull_bone = next_bone;
            }

            // generate the points in memory for this object
            int start_point = -1;
            int end_point = -1;

            int[] convex_table = new int[new_hull.length];
            convex_buffer.clear();

            // create convex hull points first, in hull order (not mesh order)
            for (int point_index = 0; point_index < new_hull.length; point_index++)
            {
                var next_vertex = new_hull[point_index];
                var new_point = CLUtils.arg_float2(next_vertex.x(), next_vertex.y());

                var bone_names = next_vertex.bone_names();
                int[] bone_ids = new int[4];
                for (int i = 0; i < bone_ids.length; i++)
                {
                    bone_ids[i] = find_bone_index(bone_map, bone_names, i);
                }
                var next_point = world.create_point(new_point, bone_ids, next_vertex.index(), next_hull, 0,0);

                if (start_point == -1)
                {
                    start_point = next_point;
                }
                end_point = next_point;

                convex_table[point_index] = next_point;
                convex_buffer.add(new_point);
            }

            // any interior points are added after convex points. Points retain their original
            // reference vertex ID, allowing them to be accessed in mesh-order when necessary
            for (Vertex next_vertex : new_interior_hull)
            {
                var new_point = CLUtils.arg_float2(next_vertex.x(), next_vertex.y());
                var bone_names = next_vertex.bone_names();
                int[] bone_ids = new int[4];
                for (int i = 0; i < bone_ids.length; i++)
                {
                    bone_ids[i] = find_bone_index(bone_map, bone_names, i);
                }
                end_point = world.create_point(new_point, bone_ids, next_vertex.index(), next_hull, 0, PointFlags.IS_INTERIOR.bits);
            }

            // generate edges in memory for this object
            int edge_start = -1;
            int edge_end = -1;
            for (int point_1_index = 0; point_1_index < new_hull.length; point_1_index++)
            {
                int point_2_index = point_1_index + 1;
                if (point_2_index == new_hull.length)
                {
                    point_2_index = 0;
                }
                var point_1 = convex_buffer.get(point_1_index);
                var point_2 = convex_buffer.get(point_2_index);
                var distance = edgeDistance(point_2, point_1);
                var next_edge = world.create_edge(convex_table[point_1_index], convex_table[point_2_index], distance, 0);
                if (edge_start == -1)
                {
                    edge_start = next_edge;
                }
                edge_end = next_edge;
            }

            // calculate interior edges

            // connect every other
            if (new_hull.length > 4)
            {
                //pass 1
                for (int p1_index = 0; p1_index < new_hull.length; p1_index++)
                {
                    int p2_index = p1_index + 2;
                    if (p2_index > convex_buffer.size() - 1)
                    {
                        continue;
                    }
                    var p1 = convex_buffer.get(p1_index);
                    var p2 = convex_buffer.get(p2_index);
                    var distance = edgeDistance(p2, p1);
                    edge_end = world.create_edge(convex_table[p1_index], convex_table[p2_index], distance, EdgeFlags.IS_INTERIOR.bits);
                }
            }

            // pass 2
            boolean odd_count = new_hull.length % 2 != 0;
            int half_count = new_hull.length / 2;
            int quarter_count = half_count / 2;
            for (int p1_index = 0; p1_index < half_count; p1_index++)
            {
                int p2_index = p1_index + half_count;
                var p1 = convex_buffer.get(p1_index);
                var p2 = convex_buffer.get(p2_index);
                var distance = edgeDistance(p2, p1);
                edge_end = world.create_edge(convex_table[p1_index], convex_table[p2_index], distance, EdgeFlags.IS_INTERIOR.bits);

                if (quarter_count > 1)
                {
                    int p3_index = p1_index + quarter_count;
                    var p3 = convex_buffer.get(p3_index);
                    var distance2 = edgeDistance(p3, p1);
                    edge_end = world.create_edge(convex_table[p1_index], convex_table[p3_index], distance2, EdgeFlags.IS_INTERIOR.bits);
                }
            }
            if (odd_count) // if there was an odd vertex at the end, connect it to the mid-point
            {
                int p2_index = convex_table.length - 1;
                var p1 = convex_buffer.get(half_count + 1);
                var p2 = convex_buffer.get(p2_index);
                var distance = edgeDistance(p2, p1);
                edge_end = world.create_edge(convex_table[half_count + 1], convex_table[p2_index], distance, EdgeFlags.IS_INTERIOR.bits);
            }

            // calculate centroid and reference angle
            MathEX.centroid(vector_buffer, new_hull);
            var l1 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, vector_buffer.x, vector_buffer.y + 1);
            var l2 = CLUtils.arg_float4(vector_buffer.x, vector_buffer.y, new_hull[0].x(), new_hull[0].y());
            var angle = MathEX.angle_between_lines(l1, l2);

            var point_table = CLUtils.arg_int2(start_point, end_point);
            var edge_table = CLUtils.arg_int2(edge_start, edge_end);
            var position = CLUtils.arg_float2(vector_buffer.x, vector_buffer.y);
            var scale = CLUtils.arg_float2(size, size);
            var rotation = CLUtils.arg_float2(0, angle);

            int flag_bits = HullFlags.IS_POLYGON._int | local_hull_flags;
            int[] bone_table = CLUtils.arg_int2(start_hull_bone, end_hull_bone);
            int hull_id = world.create_hull(hull_mesh.mesh_id(),
                position,
                scale,
                rotation,
                point_table,
                edge_table,
                bone_table,
                friction,
                restitution,
                next_entity_id,
                uv_offset,
                flag_bits);

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
            }
        }

        if (root_hull_id == -1)
        {
            throw new IllegalStateException("There was no root hull determined. "
                + "Check model data to ensure it is correct");
        }

        int[] hull_table = CLUtils.arg_int2(first_hull, last_hull);
        int[] bone_table = CLUtils.arg_int2(first_armature_bone, last_armature_bone);

        int idle_animation_id = AnimationState.IDLE.ordinal();

        int[] result = new int[2];
        result[0] = world.create_entity(x, y, x, y,
            hull_table, bone_table,
            mass, idle_animation_id, 0.0f,
            root_hull_id, model_index, model.root_transform_index(),
            -1,
            flags);
        result[1] = root_hull_id;
        return result;
    }

    private static final int[] EMPTY_TABLE = new int[]{ 0, -1 };

    private static int[] make_table(int[] ids)
    {
        return ids.length == 0
            ? EMPTY_TABLE
            : new int[]{ ids[0], ids[ids.length - 1] };
    }

    private static int get_index(int offset, int[] ids)
    {
        return offset == -1
            ? -1
            : ids[offset];
    }

    public static void load_entity(SectorContainer world, UnloadedEntity entity)
    {
        int eh_index = 0;
        int eb_index = 0;
        int[] eh_ids = new int[entity.hulls().length];
        int[] eb_ids = new int[entity.bones().length];
        for (var bone : entity.bones())
        {
            int parent_id = get_index(bone.bone_parent(), eb_ids);
            eb_ids[eb_index++] = world.create_entity_bone(bone.bone_reference(), parent_id, bone.bone());
        }
        for (var hull : entity.hulls())
        {
            int hp_index = 0;
            int he_index = 0;
            int hb_index = 0;
            int[] hp_ids = new int[hull.points().length];
            int[] he_ids = new int[hull.edges().length];
            int[] hb_ids = new int[hull.bones().length];
            for (var bone : hull.bones())
            {
                int bind_id = eb_ids[bone.bind_id()];
                hb_ids[hb_index++] = world.create_hull_bone(bone.bone_data(), bind_id, bone.inv_bind_id());
            }
            for (var point : hull.points())
            {
                int[] bone_table = new int[]
                    {
                        get_index(point.bone_1(), hb_ids),
                        get_index(point.bone_2(), hb_ids),
                        get_index(point.bone_3(), hb_ids),
                        get_index(point.bone_4(), hb_ids),
                    };
                float[] position = new float[]{ point.x(), point.y(), point.z(), point.w()};
                hp_ids[hp_index++] = world.create_point(position, bone_table, point.vertex_reference(),
                    world.next_hull(), point.hit_count(), point.flags());
            }
            for (var edge : hull.edges())
            {
                he_ids[he_index++] = world.create_edge(hp_ids[edge.p1()], hp_ids[edge.p2()], edge.length(), edge.flags());
            }

            int[] hp_tbl = make_table(hp_ids);
            int[] he_tbl = make_table(he_ids);
            int[] hb_tbl = make_table(hb_ids);
            float[] pos = new float[]{ hull.x(), hull.y(), hull.z(), hull.w() };
            float[] scl = new float[]{ hull.scale_x(), hull.scale_y() };
            float[] rot = new float[]{ hull.rotation_x(), hull.rotation_y() };
            eh_ids[eh_index++] = world.create_hull(hull.mesh_id(), pos, scl, rot,
                hp_tbl, he_tbl, hb_tbl, hull.friction(), hull.restitution(),
                world.next_entity(), hull.uv_offset(), hull.flags());
        }

        int[] eh_tbl = make_table(eh_ids);
        int[] eb_tbl = make_table(eb_ids);
        world.create_entity(entity.x(), entity.y(), entity.z(), entity.w(),
            eh_tbl, eb_tbl, entity.mass(), entity.anim_index_x(), entity.anim_elapsed_x(),
            eh_ids[entity.root_hull()], entity.model_id(), entity.model_transform_id(), entity.type(), entity.flags());
    }

    private static int find_bone_index(Map<String, Integer> bone_map, String[] bone_names, int index)
    {
        return bone_names[index] == null
            ? -1
            : bone_map.get(bone_names[index]);
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
            output[i] = new Vertex(next.index(), vec.x, vec.y, next.uv_data(), next.bone_names(), next.bone_weights());
        }
        return output;
    }


    public static Vertex[] generate_convex_hull(Mesh mesh, Vertex[] source)
    {
        var out = new Vertex[mesh.hull().length];
        for (int i = 0; i < mesh.hull().length; i++)
        {
            var next_index = mesh.hull()[i];
            out[i] = source[next_index];
        }
        return out;
    }

    public static Vertex[] generate_interior_hull(Mesh mesh, Vertex[] source)
    {
        int cx = source.length - mesh.hull().length;
        if (cx <= 0)
        {
            return new Vertex[0];
        }
        var out = new Vertex[cx];

        Set<Integer> conv = new HashSet<>();
        for (int i = 0; i < mesh.hull().length; i++)
        {
            conv.add(mesh.hull()[i]);
        }

        int c = 0;
        for (int i = 0; i < source.length; i++)
        {
            if (!conv.contains(i))
            {
                out[c++] = source[i];
            }
        }
        return out;
    }


    public static int orientation(Vertex p, Vertex q, Vertex r)
    {
        float val = (q.y() - p.y()) * (r.x() - q.x()) -
            (q.x() - p.x()) * (r.y() - q.y());

        if (val == 0)
        {
            return 0;  // collinear
        }
        return (val > 0)
            ? 1
            : 2; // clockwise or counter-clockwise
    }


    public static Vertex[] calculate_convex_hull(Vertex[] in_points)
    {
        Vertex[] points = new Vertex[in_points.length];
        System.arraycopy(in_points, 0, points, 0, in_points.length);
        int n = in_points.length;

        // There must be at least 3 points, or the hull is convex by default
        if (n < 3)
        {
            return points;
        }

        // during hull creation, this holds the vertices that are currently designated as the convex hull
        hull_vertex_buffer.clear();

        // Find the leftmost point
        int l = 0;
        for (int i = 1; i < n; i++)
        {
            if (points[i].x() < points[l].x())
            {
                l = i;
            }
        }

        // from leftmost point, move counterclockwise until the start point is reached again.
        int p = l, q;
        do
        {
            // Add current point to result
            hull_vertex_buffer.push(points[p]);

            // Search for a point 'q' such that
            // orientation(p, q, x) is counterclockwise
            // for all points 'x'. The idea is to keep
            // track of last visited most counterclock-
            // wise point in q. If any point 'i' is more
            // counterclock-wise than q, then update q.
            q = (p + 1) % n;

            for (int i = 0; i < n; i++)
            {
                // If i is more counterclockwise than
                // current q, then update q
                if (orientation(points[p], points[i], points[q])
                    == 2)
                {
                    q = i;
                }
            }

            // Now q is the most counterclockwise with
            // respect to p. Set p as q for next iteration,
            // so that q is added to result 'hull'
            p = q;

        } while (p != l);  // While we don't come to first
        // point

        return hull_vertex_buffer.toArray(Vertex[]::new);
    }

    /**
     * Calculate a convex hull for the provided vertices, but instead of returning the vertices directly, an
     * index array is returned. This array contains a mapping for each hull vertex to the corresponding vertex
     * in the input vertex array.
     * This is useful for storing the hull data in a more compact format, because the hull is itself made up of
     * vertices that are already present in the meshe data, it is more space efficient to store them as an
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
}
