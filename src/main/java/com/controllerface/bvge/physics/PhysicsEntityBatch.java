package com.controllerface.bvge.physics;

import java.util.ArrayList;
import java.util.List;

public class PhysicsEntityBatch
{
    private int point_index = 0;
    private int edge_index = 0;
    private int hull_index = 0;
    private int entity_index = 0;
    private int armature_bone_index = 0;
    private int hull_bone_index = 0;

    public final List<ArmatureBone> armature_bones = new ArrayList<>();
    public final List<HullBone> hull_bones = new ArrayList<>();
    public final List<Point> points = new ArrayList<>();
    public final List<Edge> edges = new ArrayList<>();
    public final List<Hull> hulls = new ArrayList<>();
    public final List<Entity> entities = new ArrayList<>();

    public record ArmatureBone(int bone_reference, int bone_parent_id, float[] bone_data) { }
    public record HullBone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id) { }
    public record Point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int flags) { }
    public record Edge(int p1, int p2, float l, int flags) { }

    public record Hull(int mesh_id,
                        float[] position,
                        float[] scale,
                        float[] rotation,
                        int[] point_table,
                        int[] edge_table,
                        int[] bone_table,
                        float friction,
                        float restitution,
                        int entity_id,
                        int uv_offset, int flags) { }

    public record Entity(float x,
                          float y,
                          int[] hull_table,
                          int[] bone_table,
                          float mass,
                          int anim_index,
                          float anim_time,
                          int root_hull,
                          int model_id,
                          int model_transform_id, int flags) { }


    public int new_armature_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        armature_bones.add(new ArmatureBone(bone_reference, bone_parent_id, bone_data));
        return armature_bone_index++;
    }

    public int new_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        hull_bones.add(new HullBone(bone_data, bind_pose_id, inv_bind_pose_id));
        return hull_bone_index++;
    }

    public int new_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int flags)
    {
        points.add(new Point(position, bone_ids, vertex_index, hull_index, flags));
        return point_index++;
    }

    public int new_edge(int p1, int p2, float l, int flags)
    {
        edges.add(new Edge(p1, p2, l, flags));
        return edge_index++;
    }

    public int new_hull(int mesh_id,
                        float[] position,
                        float[] scale,
                        float[] rotation,
                        int[] point_table,
                        int[] edge_table,
                        int[] bone_table,
                        float friction,
                        float restitution,
                        int entity_id,
                        int uv_offset,
                        int flags)
    {
        hulls.add(new Hull(mesh_id, position, scale, rotation, point_table, edge_table, bone_table, friction, restitution, entity_id, uv_offset, flags));
        return hull_index++;
    }

    public int new_entity(float x, float y,
                          int[] hull_table,
                          int[] bone_table,
                          float mass,
                          int anim_index,
                          float anim_time,
                          int root_hull,
                          int model_id,
                          int model_transform_id,
                          int flags)
    {
        entities.add(new Entity(x, y, hull_table, bone_table, mass, anim_index, anim_time, root_hull, model_id, model_transform_id, flags));
        return entity_index++;
    }

    public int next_entity()
    {
        return entity_index;
    }

    public int next_hull()
    {
        return hull_index;
    }

}
