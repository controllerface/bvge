/**
This is a collection of Create/Read/Update/Delete (CRUD) functions that are used
to query and update objects stored on the GPU. Unlike most kernels, these functions
are designed to operate on a single target object. 
 */

// create functions

__kernel void create_point(__global float4 *points,
                           __global int *point_vertex_references,
                           __global int *point_hull_indices,
                           __global ushort *point_hit_counts,
                           __global int *point_flags,
                           __global int4 *point_bone_tables,
                           int target,
                           float4 new_point,
                           int new_point_vertex_reference,
                           int new_point_hull_index,
                           ushort new_point_hit_count,
                           int new_point_flags,
                           int4 new_bone_table)
{
    points[target]                  = new_point; 
    point_vertex_references[target] = new_point_vertex_reference; 
    point_hull_indices[target]      = new_point_hull_index; 
    point_flags[target]             = new_point_flags; 
    point_bone_tables[target]             = new_bone_table; 
    point_hit_counts[target]        = new_point_hit_count;
}

__kernel void create_edge(__global int2 *edges,
                          __global float *edge_lengths,
                          __global int *edge_flags,
                          int target,
                          int2 new_edge,
                          float new_edge_length,
                          int new_edge_flag)
{
    edges[target]        = new_edge; 
    edge_lengths[target] = new_edge_length; 
    edge_flags[target]   = new_edge_flag; 
}

__kernel void create_bone_channel(__global int *animation_timing_indices,
                                  __global int2 *bone_pos_channel_tables,
                                  __global int2 *bone_rot_channel_tables,
                                  __global int2 *bone_scl_channel_tables,
                                  int target,
                                  int new_animation_timing_index,
                                  int2 new_bone_pos_channel_table,
                                  int2 new_bone_rot_channel_table,
                                  int2 new_bone_scl_channel_table)
{
    animation_timing_indices[target] = new_animation_timing_index;
    bone_pos_channel_tables[target]  = new_bone_pos_channel_table;
    bone_rot_channel_tables[target]  = new_bone_rot_channel_table;
    bone_scl_channel_tables[target]  = new_bone_scl_channel_table;
}

__kernel void set_bone_channel_table(__global int2 *bone_channel_tables,
                                     int target,
                                     int2 new_bone_channel_table)
{
    bone_channel_tables[target] = new_bone_channel_table;
}

__kernel void create_animation_timings(__global float *animation_durations,
                                       __global float *animation_tick_rates,
                                       int target,
                                       float new_animation_duration,
                                       float new_animation_tick_rate)
{
    animation_durations[target]  = new_animation_duration;
    animation_tick_rates[target] = new_animation_tick_rate;
}

__kernel void create_keyframe(__global float4 *key_frames,
                              __global float *frame_times,
                              int target,
                              float4 new_keyframe,
                              float new_frame_time)
{
    key_frames[target]  = new_keyframe;
    frame_times[target] = new_frame_time;
}

__kernel void create_texture_uv(__global float2 *texture_uvs,
                                int target,
                                float2 new_texture_uv)
{
    texture_uvs[target] = new_texture_uv; 
}

__kernel void create_entity(__global float4 *entities,
                            __global float2 *entity_animation_elapsed,
                            __global short2 *entity_motion_states,
                            __global int2 *entity_animation_indices,
                            __global int2 *entity_hull_tables,
                            __global int2 *entity_bone_tables,
                            __global float *entity_masses,
                            __global int *entity_root_hulls,
                            __global int *entity_model_indices,
                            __global int *entity_model_transforms,
                            __global int *entity_flags,
                            int target,
                            float4 new_entity,
                            float2 new_entity_animation_time,
                            short2 new_entity_motion_state,
                            int2 new_entity_animation_index,
                            int2 new_entity_hull_table,
                            int2 new_entity_bone_table,
                            float new_entity_mass,
                            int new_entity_root_hull,
                            int new_entity_model_id,
                            int new_entity_model_transform,
                            int new_entity_flags)
{
    entities[target]                 = new_entity; 
    entity_root_hulls[target]        = new_entity_root_hull; 
    entity_model_indices[target]     = new_entity_model_id; 
    entity_model_transforms[target]  = new_entity_model_transform; 
    entity_flags[target]             = new_entity_flags; 
    entity_hull_tables[target]       = new_entity_hull_table;
    entity_bone_tables[target]       = new_entity_bone_table; 
    entity_masses[target]            = new_entity_mass;
    entity_animation_indices[target] = new_entity_animation_index; 
    entity_animation_elapsed[target] = new_entity_animation_time;
    entity_motion_states[target]     = new_entity_motion_state;
}

__kernel void create_vertex_reference(__global float2 *vertex_references,
                                      __global float4 *vertex_weights,
                                      __global int2 *uv_tables,
                                      int target,
                                      float2 new_vertex_reference,
                                      float4 new_vertex_weights,
                                      int2 new_uv_table)
{
    vertex_references[target] = new_vertex_reference; 
    vertex_weights[target]    = new_vertex_weights; 
    uv_tables[target]         = new_uv_table;
}

__kernel void create_model_transform(__global float16 *model_transforms,
                                     int target,
                                     float16 new_model_transform)
{
    model_transforms[target] = new_model_transform; 
}

__kernel void create_bone_bind_pose(__global float16 *bone_bind_poses,
                                    int target,
                                    float16 new_bone_bind_pose)
{
    bone_bind_poses[target] = new_bone_bind_pose; 
}

__kernel void create_bone_reference(__global float16 *bone_references,
                                    int target,
                                    float16 new_bone_reference)
{
    bone_references[target] = new_bone_reference; 
}

__kernel void create_hull_bone(__global float16 *hull_bones,
                               __global int *hull_bind_pose_indicies,
                               __global int *hull_inv_bind_pose_indicies,
                               int target,
                               float16 new_hull_bone,
                               int new_hull_bind_pose_id,
                               int new_hull_inv_bind_pose_id)
{
    hull_bones[target]                  = new_hull_bone; 
    hull_bind_pose_indicies[target]     = new_hull_bind_pose_id; 
    hull_inv_bind_pose_indicies[target] = new_hull_inv_bind_pose_id; 
}

__kernel void create_armature_bone(__global float16 *armature_bones,
                                   __global int *armature_bone_reference_ids,
                                   __global int *armature_bone_parent_ids,
                                   int target,
                                   float16 new_armature_bone,
                                   int new_armature_bone_reference,
                                   int new_armature_bone_parent_id)
{
    armature_bones[target]              = new_armature_bone; 
    armature_bone_reference_ids[target] = new_armature_bone_reference;
    armature_bone_parent_ids[target]    = new_armature_bone_parent_id;
}

__kernel void create_mesh_reference(__global int2 *mesh_vertex_tables,
                                    __global int2 *mesh_face_tables,
                                    int target,
                                    int2 new_mesh_vertex_table,
                                    int2 new_mesh_face_table)
{
    mesh_vertex_tables[target] = new_mesh_vertex_table;
    mesh_face_tables[target]   = new_mesh_face_table;
}

__kernel void create_mesh_face(__global int4 *mesh_faces,
                               int target,
                               int4 new_mesh_face)
{
    mesh_faces[target] = new_mesh_face;
}

__kernel void create_hull(__global float4 *hulls,
                          __global float2 *hull_scales,
                          __global float2 *hull_rotations,
                          __global float *hull_frictions,
                          __global float *hull_restitutions,
                          __global int2 *hull_point_tables,
                          __global int2 *hull_edge_tables,
                          __global int2 *bone_tables,
                          __global int *hull_entity_ids,
                          __global int *hull_flags,
                          __global int *hull_mesh_ids,
                          __global int *hull_uv_offsets,
                          __global int *hull_integrity,
                          int target,
                          float4 new_hull,
                          float2 new_hull_scale,
                          float2 new_rotation,
                          float new_friction,
                          float new_restitution,
                          int2 new_point_table,
                          int2 new_edge_table,
                          int2 new_bone_table,
                          int new_entity_id,
                          int new_flags,
                          int new_hull_mesh_id,
                          int new_hull_uv_offset,
                          int new_hull_integrity)
{
    hulls[target]             = new_hull; 
    hull_scales[target]       = new_hull_scale; 
    hull_rotations[target]    = new_rotation; 
    hull_frictions[target]    = new_friction;
    hull_restitutions[target] = new_restitution; 
    hull_point_tables[target] = new_point_table;
    hull_edge_tables[target]  = new_edge_table;
    bone_tables[target]       = new_bone_table; 
    hull_entity_ids[target]   = new_entity_id; 
    hull_flags[target]        = new_flags; 
    hull_mesh_ids[target]     = new_hull_mesh_id;
    hull_uv_offsets[target]   = new_hull_uv_offset;
    hull_integrity[target]    = new_hull_integrity;
}

// read functions

__kernel void read_position(__global float4 *entities,
                            __global float *output,
                            int target)
{
    float4 entity = entities[target];

    output[0] = entity.x;
    output[1] = entity.y;
}

// update functions

__kernel void update_accel(__global float2 *entity_accel,
                           int target,
                           float2 new_value)
{
    float2 accel = entity_accel[target];

    accel.x = new_value.x;
    accel.y = new_value.y;

    entity_accel[target] = accel;
}

__kernel void update_mouse_position(__global int *entity_root_hulls,
                                    __global int2 *hull_point_tables,
                                    __global float4 *points,
                                    int target,
                                    float2 new_value)
{
    int h  = entity_root_hulls[target];
    int2 t = hull_point_tables[h];

    points[t.x].xy = new_value;
}

__kernel void merge_point(__global float4 *points_in,
                          __global int *point_vertex_references_in,
                          __global int *point_hull_indices_in,
                          __global ushort *point_hit_counts_in,
                          __global int *point_flags_in,
                          __global int4 *point_bone_tables_in,
                          __global float4 *points_out,
                          __global int *point_vertex_references_out,
                          __global int *point_hull_indices_out,
                          __global ushort *point_hit_counts_out,
                          __global int *point_flags_out,
                          __global int4 *point_bone_tables_out,
                          int point_offset,
                          int bone_offset,
                          int hull_offset)
{
    int current_point = get_global_id(0);
    int target_point = current_point + point_offset;
    points_out[target_point]                  = points_in[current_point]; 
    point_vertex_references_out[target_point] = point_vertex_references_in[current_point]; 
    point_hull_indices_out[target_point]      = point_hull_indices_in[current_point] + hull_offset; 
    point_hit_counts_out[target_point]        = point_hit_counts_in[current_point]; 
    point_flags_out[target_point]             = point_flags_in[current_point]; 
    point_bone_tables_out[target_point]       = point_bone_tables_in[current_point] + (int4)(bone_offset); 
}

__kernel void merge_edge(__global int2 *edges_in,
                          __global float *edge_lengths_in,
                          __global int *edge_flags_in,
                          __global int2 *edges_out,
                          __global float *edge_lengths_out,
                          __global int *edge_flags_out,
                          int edge_offset,
                          int point_offset)
{
    int current_edge = get_global_id(0);
    int target_edge = current_edge + edge_offset;
    edges_out[target_edge]        = edges_in[current_edge] + (int2)(point_offset); 
    edge_lengths_out[target_edge] = edge_lengths_in[current_edge]; 
    edge_flags_out[target_edge]   = edge_flags_in[current_edge]; 
}

__kernel void merge_hull_bone(__global float16 *hull_bones_in,
                               __global int *hull_bind_pose_indicies_in,
                               __global int *hull_inv_bind_pose_indicies_in,
                               __global float16 *hull_bones_out,
                               __global int *hull_bind_pose_indicies_out,
                               __global int *hull_inv_bind_pose_indicies_out,
                               int hull_bone_offset,
                               int armature_bone_offset)
{
    int current_hull_bone = get_global_id(0);
    int target_hull_bone = current_hull_bone + hull_bone_offset;
    hull_bones_out[target_hull_bone]                  = hull_bones_in[current_hull_bone]; 
    hull_bind_pose_indicies_out[target_hull_bone]     = hull_bind_pose_indicies_in[current_hull_bone] + armature_bone_offset; 
    hull_inv_bind_pose_indicies_out[target_hull_bone] = hull_inv_bind_pose_indicies_in[current_hull_bone]; 
}

__kernel void merge_armature_bone(__global float16 *armature_bones_in,
                                   __global int *armature_bone_reference_ids_in,
                                   __global int *armature_bone_parent_ids_in,
                                   __global float16 *armature_bones_out,
                                   __global int *armature_bone_reference_ids_out,
                                   __global int *armature_bone_parent_ids_out,
                                   int armature_bone_offset)
{
    int current_armature_bone = get_global_id(0);
    int target_armature_bone = current_armature_bone + armature_bone_offset;
    armature_bones_out[target_armature_bone]              = armature_bones_in[current_armature_bone]; 
    armature_bone_reference_ids_out[target_armature_bone] = armature_bone_reference_ids_in[current_armature_bone];
    armature_bone_parent_ids_out[target_armature_bone]    = armature_bone_parent_ids_in[current_armature_bone] + armature_bone_offset;
}

__kernel void merge_hull(__global float4 *hulls_in,
                          __global float2 *hull_scales_in,
                          __global float2 *hull_rotations_in,
                          __global float *hull_frictions_in,
                          __global float *hull_restitutions_in,
                          __global int2 *hull_point_tables_in,
                          __global int2 *hull_edge_tables_in,
                          __global int2 *bone_tables_in,
                          __global int *hull_entity_ids_in,
                          __global int *hull_flags_in,
                          __global int *hull_mesh_ids_in,
                          __global int *hull_uv_offsets_in,
                          __global int *hull_integrity_in,
                          __global float4 *hulls_out,
                          __global float2 *hull_scales_out,
                          __global float2 *hull_rotations_out,
                          __global float *hull_frictions_out,
                          __global float *hull_restitutions_out,
                          __global int2 *hull_point_tables_out,
                          __global int2 *hull_edge_tables_out,
                          __global int2 *bone_tables_out,
                          __global int *hull_entity_ids_out,
                          __global int *hull_flags_out,
                          __global int *hull_mesh_ids_out,
                          __global int *hull_uv_offsets_out,
                          __global int *hull_integrity_out,
                          int hull_offset,
                          int hull_bone_offset,
                          int entity_offset,
                          int edge_offset,
                          int point_offset)
{
    int current_hull = get_global_id(0);
    int target_hull = current_hull + hull_offset;
    
    hulls_out[target_hull]             = hulls_in[current_hull];
    hull_scales_out[target_hull]       = hull_scales_in[current_hull];
    hull_rotations_out[target_hull]    = hull_rotations_in[current_hull];
    hull_frictions_out[target_hull]    = hull_frictions_in[current_hull];
    hull_restitutions_out[target_hull] = hull_restitutions_in[current_hull];
    hull_point_tables_out[target_hull] = hull_point_tables_in[current_hull] + (int2)(point_offset);
    hull_edge_tables_out[target_hull]  = hull_edge_tables_in[current_hull] + (int2)(edge_offset);
    bone_tables_out[target_hull]       = bone_tables_in[current_hull] + (int2)(hull_bone_offset);
    hull_entity_ids_out[target_hull]   = hull_entity_ids_in[current_hull] + entity_offset;
    hull_flags_out[target_hull]        = hull_flags_in[current_hull];
    hull_mesh_ids_out[target_hull]     = hull_mesh_ids_in[current_hull];
    hull_uv_offsets_out[target_hull]   = hull_uv_offsets_in[current_hull];
    hull_integrity_out[target_hull]    = hull_integrity_in[current_hull];
}

__kernel void merge_entity(__global float4 *entities_in,
                            __global float2 *entity_animation_elapsed_in,
                            __global short2 *entity_motion_states_in,
                            __global int2 *entity_animation_indices_in,
                            __global int2 *entity_hull_tables_in,
                            __global int2 *entity_bone_tables_in,
                            __global float *entity_masses_in,
                            __global int *entity_root_hulls_in,
                            __global int *entity_model_indices_in,
                            __global int *entity_model_transforms_in,
                            __global int *entity_flags_in,
                            __global float4 *entities_out,
                            __global float2 *entity_animation_elapsed_out,
                            __global short2 *entity_motion_states_out,
                            __global int2 *entity_animation_indices_out,
                            __global int2 *entity_hull_tables_out,
                            __global int2 *entity_bone_tables_out,
                            __global float *entity_masses_out,
                            __global int *entity_root_hulls_out,
                            __global int *entity_model_indices_out,
                            __global int *entity_model_transforms_out,
                            __global int *entity_flags_out,
                            int entity_offset,
                            int hull_offset,
                            int armature_bone_offset)
{
    int current_entity = get_global_id(0);
    int target_entity = current_entity + entity_offset;

    entities_out[target_entity]                 = entities_in[current_entity];
    entity_animation_elapsed_out[target_entity] = entity_animation_elapsed_in[current_entity];
    entity_motion_states_out[target_entity]     = entity_motion_states_in[current_entity];
    entity_animation_indices_out[target_entity] = entity_animation_indices_in[current_entity];
    entity_hull_tables_out[target_entity]       = entity_hull_tables_in[current_entity] + (int2)(hull_offset);
    entity_bone_tables_out[target_entity]       = entity_bone_tables_in[current_entity] + (int2)(armature_bone_offset);
    entity_masses_out[target_entity]            = entity_masses_in[current_entity];
    entity_root_hulls_out[target_entity]        = entity_root_hulls_in[current_entity] + hull_offset;
    entity_model_indices_out[target_entity]     = entity_model_indices_in[current_entity];
    entity_model_transforms_out[target_entity]  = entity_model_transforms_in[current_entity];
    entity_flags_out[target_entity]             = entity_flags_in[current_entity];
}

__kernel void count_egress_entities(__global int *entity_flags,
                                    __global int2 *entity_hull_tables,
                                    __global int2 *entity_bone_tables,
                                    __global int2 *hull_point_tables,
                                    __global int2 *hull_edge_tables,
                                    __global int2 *hull_bone_tables,
                                    __global int *counter)
{
    int current_entity = get_global_id(0);

    int flags       = entity_flags[current_entity];
    bool sector_out = (flags & SECTOR_OUT) !=0;
    bool deleted    = (flags & DELETED) !=0;

    if (deleted) atomic_inc(&counter[6]);
    if (sector_out)
    {
        int2 hull_table         = entity_hull_tables[current_entity];
        int2 entitiy_bone_table = entity_bone_tables[current_entity];
        
        int hull_count         = hull_table.y - hull_table.x + 1;
        int entitiy_bone_count = entitiy_bone_table.y - entitiy_bone_table.x + 1;
        int point_count        = 0;
        int edge_count         = 0;
        int hull_bone_count    = 0;

        for (int current_hull = hull_table.x; current_hull <= hull_table.y; current_hull++)
        {
            int2 point_table     = hull_point_tables[current_hull];
            int2 edge_table      = hull_edge_tables[current_hull];
            int2 hull_bone_table = hull_bone_tables[current_hull];

            point_count     += point_table.y - point_table.x + 1;
            edge_count      += edge_table.y - edge_table.x + 1;
            hull_bone_count += hull_bone_table.y - hull_bone_table.x + 1;
        }

        atomic_inc(&counter[0]); 
        atomic_add(&counter[1], hull_count);
        atomic_add(&counter[2], point_count);
        atomic_add(&counter[3], edge_count);
        atomic_add(&counter[4], hull_bone_count);
        atomic_add(&counter[5], entitiy_bone_count);
        
        flags = (flags | DELETED);
        entity_flags[current_entity] = flags;
    }
}



// todo: implement for armature
__kernel void rotate_hull(__global float4 *hulls,
                          __global int4 *element_tables,
                          __global float4 *points,
                          int target,
                          float angle)
{
    float4 hull        = hulls[target];
    int4 element_table = element_tables[target];
    int start          = element_table.x;
    int end            = element_table.y;
    float2 origin      = (float2)(hull.x, hull.y);
    for (int i = start; i <= end; i++)
    {
        float4 point = points[i];
        points[i] = rotate_point(point, origin, angle);
    }
}
