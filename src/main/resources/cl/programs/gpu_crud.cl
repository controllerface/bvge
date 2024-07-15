/**
This is a collection of Create/Read/Update/Delete (CRUD) functions that are used
to query and update objects stored on the GPU. Unlike most kernels, these functions
are designed to operate on a single target object. 
 */

// read functions

__kernel void read_position(__global float4 *entities,
                            __global float *output,
                            int target)
{
    float4 entity = entities[target];

    output[0] = entity.x;
    output[1] = entity.y;
}

__kernel void read_entity_info(__global float4 *entities,
                              __global float2 *entity_accel,
                              __global short2 *entity_motion_states,
                              __global int *entity_flags,
                              __global int2 *entity_animation_layers,
                              __global int2 *entity_animation_previous,
                              __global float2 *entity_animation_elapsed,
                              __global float2 *entity_animation_blend,
                              __global float *output,
                              int target)
{
    
    float4 entity            = entities[target];
    float2 accel             = entity_accel[target];
    float2 current_time      = entity_animation_elapsed[target];
    float2 current_blend     = entity_animation_blend[target];
    short2 motion_state      = entity_motion_states[target];
    int2 anim_layers         = entity_animation_layers[target];
    int2 anim_previous       = entity_animation_previous[target];
    int arm_flag             = entity_flags[target];

    // printf("debug out: [ex: %f ey: %f ez: %f ew: %f, acc.x: %f acc.y: %f, t.x: %f t.y: %f, blend.x: %f blend.y: %f, motion.x: %d motion.y: %d, anim.x: %d anim.x: %d, flag: %d]\n",
    //     entity.x, entity.y, entity.z, entity.w,
    //     accel.x, accel.y,
    //     current_time.x, current_time.y,
    //     current_blend.x, current_blend.y,
    //     motion_state.x,
    //     motion_state.x,
    //     anim_layers.x,
    //     anim_layers.x,
    //     arm_flag);

    output[0] = entity.x;
    output[1] = entity.y;
    output[2] = entity.z;
    output[3] = entity.w;
    output[4] = accel.x;
    output[5] = accel.y;
    output[6] = current_time.x;
    output[7] = current_time.y;
    output[8] = current_blend.x;
    output[9] = current_blend.y;

    output[10] = (float)motion_state.x;
    output[11] = (float)motion_state.y;
    output[12] = (float)anim_layers.x;
    output[13] = (float)anim_layers.y;
    output[14] = (float)anim_previous.x;
    output[15] = (float)anim_previous.y;
    output[16] = (float)arm_flag;
}










__kernel void write_entity_info(__global float2 *entity_accel,
                              __global float2 *entity_animation_elapsed,
                              __global float2 *entity_animation_blend,
                              __global short2 *entity_motion_states,
                              __global int2 *entity_animation_layers,
                              __global int2 *entity_animation_previous,
                              __global int *entity_flags,
                              int target,
                              float2 new_accel,
                              float2 new_anim_elapsed,
                              float2 new_anim_blend,
                              short2 new_motion_state,
                              int2 new_anim_layerss,
                              int2 new_anim_previous,
                              int new_flags)
{
    entity_accel[target]              = new_accel;
    entity_animation_elapsed[target]  = new_anim_elapsed;
    entity_animation_blend[target]    = new_anim_blend;
    entity_motion_states[target]      = new_motion_state;
    entity_animation_layers[target]   = new_anim_layerss;
    entity_animation_previous[target] = new_anim_previous;
    entity_flags[target]              = new_flags;
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
    int root_hull  = entity_root_hulls[target];
    int2 point_table = hull_point_tables[root_hull];
    points[point_table.x].xy = new_value;
}

__kernel void update_block_position(__global float4 *entities,
                                    __global int *entity_root_hulls,
                                    __global int2 *hull_point_tables,
                                    __global float4 *points,
                                    int target,
                                    float2 new_value)
{
    float2 center = entities[target].xy;
    int root_hull  = entity_root_hulls[target];
    int2 point_table = hull_point_tables[root_hull];
    float2 p1 = points[point_table.x].xy - center;
    float2 p2 = points[point_table.x + 1].xy - center;
    float2 p3 = points[point_table.x + 2].xy - center;
    float2 p4 = points[point_table.x + 3].xy - center;

    points[point_table.x].xy = new_value + p1;
    points[point_table.x + 1].xy = new_value + p2;
    points[point_table.x + 2].xy = new_value + p3;
    points[point_table.x + 3].xy = new_value + p4;
}

__kernel void update_select_block(__global int *entity_flags,
                                  __global int *hull_uv_offsets,
                                  __global int2 *entity_hull_tables,
                                  int target,
                                  int new_value)
{
    entity_flags[target] |= GHOST_ACTIVE;
    int2 hull_table = entity_hull_tables[target];
    hull_uv_offsets[hull_table.x] = new_value;
}

__kernel void clear_select_block(__global int *entity_flags,
                                  int target)
{
    entity_flags[target] &= ~GHOST_ACTIVE;
}

__kernel void merge_point(__global float4 *points_in,
                          __global int *point_vertex_references_in,
                          __global int *point_hull_indices_in,
                          __global short *point_hit_counts_in,
                          __global int *point_flags_in,
                          __global int4 *point_bone_tables_in,
                          __global float4 *points_out,
                          __global int *point_vertex_references_out,
                          __global int *point_hull_indices_out,
                          __global short *point_hit_counts_out,
                          __global int *point_flags_out,
                          __global int4 *point_bone_tables_out,
                          int point_offset,
                          int bone_offset,
                          int hull_offset,
                          int max_point)
{
    int current_point = get_global_id(0);

    if (current_point >= max_point) return;

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
                          int point_offset,
                          int max_edge)
{
    int current_edge = get_global_id(0);
    if (current_edge >= max_edge) return;
    int target_edge = current_edge + edge_offset;
    edges_out[target_edge]        = edges_in[current_edge] + (int2)(point_offset); 
    edge_lengths_out[target_edge] = edge_lengths_in[current_edge]; 
    edge_flags_out[target_edge]   = edge_flags_in[current_edge]; 
}

__kernel void merge_hull_bone(__global float16 *hull_bones_in,
                               __global int *hull_bind_pose_indices_in,
                               __global int *hull_inv_bind_pose_indicies_in,
                               __global float16 *hull_bones_out,
                               __global int *hull_bind_pose_indicies_out,
                               __global int *hull_inv_bind_pose_indicies_out,
                               int hull_bone_offset,
                               int armature_bone_offset,
                               int max_hull_bone)
{
    int current_hull_bone = get_global_id(0);
    if (current_hull_bone >= max_hull_bone) return;
    int target_hull_bone = current_hull_bone + hull_bone_offset;
    hull_bones_out[target_hull_bone]                  = hull_bones_in[current_hull_bone]; 
    hull_bind_pose_indicies_out[target_hull_bone]     = hull_bind_pose_indices_in[current_hull_bone] + armature_bone_offset; 
    hull_inv_bind_pose_indicies_out[target_hull_bone] = hull_inv_bind_pose_indicies_in[current_hull_bone]; 
}

__kernel void merge_entity_bone(__global float16 *entity_bones_in,
                                   __global int *entity_bone_reference_ids_in,
                                   __global int *entity_bone_parent_ids_in,
                                   __global float16 *entity_bones_out,
                                   __global int *entity_bone_reference_ids_out,
                                   __global int *entity_bone_parent_ids_out,
                                   int entity_bone_offset,
                                   int max_entity_bone)
{
    int current_entity_bone = get_global_id(0);
    if (current_entity_bone >= max_entity_bone) return;
    int target_entity_bone = current_entity_bone + entity_bone_offset;
    entity_bones_out[target_entity_bone]              = entity_bones_in[current_entity_bone]; 
    entity_bone_reference_ids_out[target_entity_bone] = entity_bone_reference_ids_in[current_entity_bone];
    entity_bone_parent_ids_out[target_entity_bone]    = entity_bone_parent_ids_in[current_entity_bone] + entity_bone_offset;
}

__kernel void merge_hull(__global float4 *hulls_in,
                          __global float2 *hull_scales_in,
                          __global float2 *hull_rotations_in,
                          __global float *hull_frictions_in,
                          __global float *hull_restitutions_in,
                          __global int2 *hull_point_tables_in,
                          __global int2 *hull_edge_tables_in,
                          __global int2 *hull_bone_tables_in,
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
                          __global int2 *hull_bone_tables_out,
                          __global int *hull_entity_ids_out,
                          __global int *hull_flags_out,
                          __global int *hull_mesh_ids_out,
                          __global int *hull_uv_offsets_out,
                          __global int *hull_integrity_out,
                          int hull_offset,
                          int hull_bone_offset,
                          int entity_offset,
                          int edge_offset,
                          int point_offset,
                          int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int target_hull = current_hull + hull_offset;
    
    hulls_out[target_hull]             = hulls_in[current_hull];
    hull_scales_out[target_hull]       = hull_scales_in[current_hull];
    hull_rotations_out[target_hull]    = hull_rotations_in[current_hull];
    hull_frictions_out[target_hull]    = hull_frictions_in[current_hull];
    hull_restitutions_out[target_hull] = hull_restitutions_in[current_hull];
    hull_point_tables_out[target_hull] = hull_point_tables_in[current_hull] + (int2)(point_offset);
    hull_edge_tables_out[target_hull]  = hull_edge_tables_in[current_hull] + (int2)(edge_offset);
    hull_bone_tables_out[target_hull]  = hull_bone_tables_in[current_hull] + (int2)(hull_bone_offset);
    hull_entity_ids_out[target_hull]   = hull_entity_ids_in[current_hull] + entity_offset;
    hull_flags_out[target_hull]        = hull_flags_in[current_hull];
    hull_mesh_ids_out[target_hull]     = hull_mesh_ids_in[current_hull];
    hull_uv_offsets_out[target_hull]   = hull_uv_offsets_in[current_hull];
    hull_integrity_out[target_hull]    = hull_integrity_in[current_hull];
}

__kernel void merge_entity(__global float4 *entities_in,
                            __global float2 *entity_animation_elapsed_in,
                            __global short2 *entity_motion_states_in,
                            __global int2 *entity_animation_layers_in,
                            __global int2 *entity_animation_previous_in,
                            __global int2 *entity_hull_tables_in,
                            __global int2 *entity_bone_tables_in,
                            __global float *entity_masses_in,
                            __global int *entity_root_hulls_in,
                            __global int *entity_model_indices_in,
                            __global int *entity_model_transforms_in,
                            __global int *entity_types_in,
                            __global int *entity_flags_in,
                            __global float4 *entities_out,
                            __global float2 *entity_animation_elapsed_out,
                            __global short2 *entity_motion_states_out,
                            __global int2 *entity_animation_layers_out,
                            __global int2 *entity_animation_previous_out,
                            __global int2 *entity_hull_tables_out,
                            __global int2 *entity_bone_tables_out,
                            __global float *entity_masses_out,
                            __global int *entity_root_hulls_out,
                            __global int *entity_model_indices_out,
                            __global int *entity_model_transforms_out,
                            __global int *entity_types_out,
                            __global int *entity_flags_out,
                            int entity_offset,
                            int hull_offset,
                            int armature_bone_offset,
                            int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    int target_entity = current_entity + entity_offset;

    entities_out[target_entity]                  = entities_in[current_entity];
    entity_animation_elapsed_out[target_entity]  = entity_animation_elapsed_in[current_entity];
    entity_motion_states_out[target_entity]      = entity_motion_states_in[current_entity];
    entity_animation_layers_out[target_entity]   = entity_animation_layers_in[current_entity];
    entity_animation_previous_out[target_entity] = entity_animation_previous_in[current_entity];
    entity_hull_tables_out[target_entity]        = entity_hull_tables_in[current_entity] + (int2)(hull_offset);
    entity_bone_tables_out[target_entity]        = entity_bone_tables_in[current_entity] + (int2)(armature_bone_offset);
    entity_masses_out[target_entity]             = entity_masses_in[current_entity];
    entity_root_hulls_out[target_entity]         = entity_root_hulls_in[current_entity] + hull_offset;
    entity_model_indices_out[target_entity]      = entity_model_indices_in[current_entity];
    entity_model_transforms_out[target_entity]   = entity_model_transforms_in[current_entity];
    entity_types_out[target_entity]              = entity_types_in[current_entity];
    entity_flags_out[target_entity]              = entity_flags_in[current_entity];
}

__kernel void count_egress_entities(__global int *entity_flags,
                                    __global int2 *entity_hull_tables,
                                    __global int2 *entity_bone_tables,
                                    __global int *hull_flags,
                                    __global int2 *hull_point_tables,
                                    __global int2 *hull_edge_tables,
                                    __global int2 *hull_bone_tables,
                                    __global int *counters,
                                    int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    int flags       = entity_flags[current_entity];
    bool sector_out = (flags & SECTOR_OUT) !=0;
    bool broken     = (flags & BROKEN) !=0;
    bool collected  = (flags & COLLECTED) !=0;

    if(collected)
    {
        flags = (flags | DELETED);
        entity_flags[current_entity] = flags;
        atomic_inc(&counters[7]); 
    }
    if(broken)
    {
        int2 hull_table = entity_hull_tables[current_entity];
        int hull_0_flags = hull_flags[hull_table.x];
        int hull_count  = hull_table.y - hull_table.x + 1;
        bool collectable = (flags & COLLECTABLE) !=0;
        flags = (flags | DELETED);
        entity_flags[current_entity] = flags;
        if (collectable) return;
        atomic_add(&counters[6], hull_count); 
    }
    if (sector_out)
    {
        int2 hull_table        = entity_hull_tables[current_entity];
        int2 entity_bone_table = entity_bone_tables[current_entity];
        
        int hull_count         = hull_table.y - hull_table.x + 1;
        int entity_bone_count  = entity_bone_table.y - entity_bone_table.x + 1;
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

        atomic_inc(&counters[0]); 
        atomic_add(&counters[1], hull_count);
        atomic_add(&counters[2], point_count);
        atomic_add(&counters[3], edge_count);
        atomic_add(&counters[4], hull_bone_count);
        atomic_add(&counters[5], entity_bone_count);
        
        flags = (flags | DELETED);
        entity_flags[current_entity] = flags;
    }
}

__kernel void egress_collected(__global int *entity_flags,
                               __global int *entity_types,
                               __global int *types,
                               __global int *counter, 
                               int max_entity)
{
    int current_entity = get_global_id(0);
    
    if (current_entity >= max_entity) return;

    int e_flags = entity_flags[current_entity];
    int e_type = entity_types[current_entity];
    bool collected = (e_flags & COLLECTED) !=0;
    if (collected)
    {
        int entity_id_offset = atomic_inc(&counter[0]); 
        types[entity_id_offset] = e_type;
    }
}

__kernel void egress_broken(__global float4 *entities, 
                            __global int *entity_flags,
                            __global int *entity_types,
                            __global int *entity_model_ids,
                            __global float2 *positions,
                            __global int *types,
                            __global int *model_ids,
                            __global int *counter, 
                            int max_entity)
{
    int current_entity = get_global_id(0);

    if (current_entity >= max_entity) return;

    int flags   = entity_flags[current_entity];
    int type    = entity_types[current_entity];
    bool broken = (flags & BROKEN) !=0;
    if (broken)
    {
        float4 entity = entities[current_entity];
        int entity_model_id = entity_model_ids[current_entity];
        bool collectable = (flags & COLLECTABLE) !=0;
        if (collectable) return;
        int entity_id_offset = atomic_inc(&counter[0]); 
        positions[entity_id_offset] = entity.xy;
        types[entity_id_offset] = type;
        model_ids[entity_id_offset] = entity_model_id;
    }
}

/**
This kernel converts entities and their constituent components from the ordered layout used in 
the game world, to an un-ordered format. During this process, all object tables are converted
from direct indices to relative offfsets. The converted objects are processed in a CPU task
that further converts them into a form that is able to then re-enter the world. At that point,
their relative offsets are mapped back to an in-order format as they are spawned. 
*/
__kernel void egress_entities(__global int *point_hull_indices_in,
                              __global int4 *point_bone_tables_in,
                              __global int2 *edges_in,
                              __global int2 *hull_point_tables_in,
                              __global int2 *hull_edge_tables_in,
                              __global int2 *hull_bone_tables_in,
                              __global int *hull_bind_pose_indices_in,
                              __global int *entity_bone_parent_ids_in,
        
                              __global float4 *entities_in,
                              __global float2 *entity_animation_elapsed_in,
                              __global short2 *entity_motion_states_in,
                              __global int2 *entity_animation_layers_in,
                              __global int2 *entity_animation_previous_in,
                              __global int2 *entity_hull_tables_in,
                              __global int2 *entity_bone_tables_in,
                              __global float *entity_masses_in,
                              __global int *entity_root_hulls_in,
                              __global int *entity_model_indices_in,
                              __global int *entity_model_transforms_in,
                              __global int *entity_types_in,
                              __global int *entity_flags_in,

                              __global float4 *entities_out,
                              __global float2 *entity_animation_elapsed_out,
                              __global short2 *entity_motion_states_out,
                              __global int2 *entity_animation_layers_out,
                              __global int2 *entity_animation_previous_out,
                              __global int2 *entity_hull_tables_out,
                              __global int2 *entity_bone_tables_out,
                              __global float *entity_masses_out,
                              __global int *entity_root_hulls_out,
                              __global int *entity_model_indices_out,
                              __global int *entity_model_transforms_out,
                              __global int *entity_types_out,
                              __global int *entity_flags_out,
 
                              __global int *new_points,
                              __global int *new_edges,
                              __global int *new_hulls,
                              __global int *new_hull_bones,
                              __global int *new_entity_bones,
                              __global int *counters, 

                              int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    int flags       = entity_flags_in[current_entity];
    bool sector_out = (flags & SECTOR_OUT) !=0;

    if (sector_out)
    {
        int2 entity_hull_table          = entity_hull_tables_in[current_entity];
        int2 entity_bone_table          = entity_bone_tables_in[current_entity];
        int entity_root_hull            = entity_root_hulls_in[current_entity];

        int root_hull_offset           = entity_root_hull - entity_hull_table.x;
        int initial_entity_bone_offset = entity_bone_table.x;

        int hull_count         = entity_hull_table.y - entity_hull_table.x + 1;
        int entity_bone_count  = entity_bone_table.y - entity_bone_table.x + 1;
        int point_count        = 0;
        int edge_count         = 0;
        int hull_bone_count    = 0;

        for (int current_hull = entity_hull_table.x; current_hull <= entity_hull_table.y; current_hull++)
        {
            int2 point_table     = hull_point_tables_in[current_hull];
            int2 edge_table      = hull_edge_tables_in[current_hull];
            int2 hull_bone_table = hull_bone_tables_in[current_hull];

            point_count     += point_table.y - point_table.x + 1;
            edge_count      += edge_table.y - edge_table.x + 1;
            hull_bone_count += hull_bone_table.y - hull_bone_table.x + 1;
        }

        int entity_id_offset   = atomic_inc(&counters[0]); 
        int hull_offset        = atomic_add(&counters[1], hull_count);
        int point_offset       = atomic_add(&counters[2], point_count);
        int edge_offset        = atomic_add(&counters[3], edge_count);
        int hull_bone_offset   = atomic_add(&counters[4], hull_bone_count);
        int entity_bone_offset = atomic_add(&counters[5], entity_bone_count);

        entities_out[entity_id_offset]                  = entities_in[current_entity];
        entity_masses_out[entity_id_offset]             = entity_masses_in[current_entity];
        entity_model_indices_out[entity_id_offset]      = entity_model_indices_in[current_entity];
        entity_model_transforms_out[entity_id_offset]   = entity_model_transforms_in[current_entity];
        entity_types_out[entity_id_offset]              = entity_types_in[current_entity];
        entity_flags_out[entity_id_offset]              = entity_flags_in[current_entity] & ~(SECTOR_OUT | DELETED);
        entity_animation_layers_out[entity_id_offset]   = entity_animation_layers_in[current_entity];
        entity_animation_previous_out[entity_id_offset] = entity_animation_previous_in[current_entity];
        entity_animation_elapsed_out[entity_id_offset]  = entity_animation_elapsed_in[current_entity];
        entity_motion_states_out[entity_id_offset]      = entity_motion_states_in[current_entity];
        entity_hull_tables_out[entity_id_offset]        = (int2)(hull_offset, hull_offset + hull_count - 1);
        entity_bone_tables_out[entity_id_offset]        = (int2)(entity_bone_offset, entity_bone_offset + entity_bone_count - 1);
        entity_root_hulls_out[entity_id_offset]         = hull_offset + root_hull_offset;

        int point_offset_count       = 0;
        int edge_offset_count        = 0;
        int hull_bone_offset_count   = 0;

        for (int current_entity_bone = entity_bone_table.x; current_entity_bone <= entity_bone_table.y; current_entity_bone++)
        {
            int new_entity_bone_id = entity_bone_offset + current_entity_bone - entity_bone_table.x;
            new_entity_bones[current_entity_bone] = new_entity_bone_id;

            int bone_parent_id = entity_bone_parent_ids_in[current_entity_bone];
            int parent_offset = bone_parent_id - initial_entity_bone_offset;
            int new_parent_id = bone_parent_id == -1 ? -1 : entity_bone_offset + parent_offset;
            entity_bone_parent_ids_in[current_entity_bone] = new_parent_id;
        }

        for (int current_hull = entity_hull_table.x; current_hull <= entity_hull_table.y; current_hull++)
        {
            int new_hull_id = hull_offset + current_hull - entity_hull_table.x;
            new_hulls[current_hull] = new_hull_id;

            int2 hull_point_table  = hull_point_tables_in[current_hull];
            int2 hull_edge_table   = hull_edge_tables_in[current_hull];
            int2 hull_bone_table   = hull_bone_tables_in[current_hull];

            int initial_hull_point_offset = hull_point_table.x;
            int initial_hull_bone_offset  = hull_bone_table.x;

            int next_hull_point_x = point_offset + point_offset_count;
            int next_hull_edge_table_x  = edge_offset + edge_offset_count;
            int next_hull_bone_table_x  = hull_bone_offset + hull_bone_offset_count;

            for (int current_hull_bone = hull_bone_table.x; current_hull_bone <= hull_bone_table.y; current_hull_bone++)
            {
                int new_hull_bone_id = hull_bone_offset + hull_bone_offset_count++;
                new_hull_bones[current_hull_bone] = new_hull_bone_id;

                int hull_bind_pose_id = hull_bind_pose_indices_in[current_hull_bone];
                int bind_pose_offset = hull_bind_pose_id - initial_entity_bone_offset;
                int new_bind_pose_offset = entity_bone_offset + bind_pose_offset;
                hull_bind_pose_indices_in[current_hull_bone] = new_bind_pose_offset;
            }

            for (int current_edge = hull_edge_table.x; current_edge <= hull_edge_table.y; current_edge++)
            {
                int new_edge_id = edge_offset + edge_offset_count++;
                new_edges[current_edge] = new_edge_id;

                int2 edge         = edges_in[current_edge];
                int2 edge_point_offsets = (int2)(0.0f, 0.0f);
                int2 new_edge_points    = (int2)(0.0f, 0.0f);
                edge_point_offsets.x = edge.x - initial_hull_point_offset;
                edge_point_offsets.y = edge.y - initial_hull_point_offset;
                new_edge_points.x = point_offset + edge_point_offsets.x;
                new_edge_points.y = point_offset + edge_point_offsets.y;
                edges_in[current_edge] = new_edge_points;
            }

            for (int current_point = hull_point_table.x; current_point <= hull_point_table.y; current_point++)
            {
                int new_point_id = point_offset + point_offset_count++;
                new_points[current_point] = new_point_id;

                int4 point_bone_table      = point_bone_tables_in[current_point];

                int4 point_bone_offsets   = (int4)(0.0f, 0.0f, 0.0f, 0.0f);
                int4 new_point_bone_table = (int4)(0.0f, 0.0f, 0.0f, 0.0f);

                point_bone_offsets.x = point_bone_table.x - initial_hull_bone_offset;
                point_bone_offsets.y = point_bone_table.y - initial_hull_bone_offset;
                point_bone_offsets.z = point_bone_table.z - initial_hull_bone_offset;
                point_bone_offsets.w = point_bone_table.w - initial_hull_bone_offset;

                new_point_bone_table.x = point_bone_table.x == -1 ? -1 : hull_bone_offset + point_bone_offsets.x;
                new_point_bone_table.y = point_bone_table.y == -1 ? -1 : hull_bone_offset + point_bone_offsets.y;
                new_point_bone_table.z = point_bone_table.z == -1 ? -1 : hull_bone_offset + point_bone_offsets.z;
                new_point_bone_table.w = point_bone_table.w == -1 ? -1 : hull_bone_offset + point_bone_offsets.w;

                point_hull_indices_in[current_point] = new_hull_id;
                point_bone_tables_in[current_point]  = new_point_bone_table;
            }

            hull_point_table.x = next_hull_point_x;
            hull_point_table.y = point_offset + point_offset_count - 1;
            hull_edge_table.x  = next_hull_edge_table_x;
            hull_edge_table.y  = edge_offset + edge_offset_count - 1;
            hull_bone_table.x  = next_hull_bone_table_x;
            hull_bone_table.y  = hull_bone_offset + hull_bone_offset_count - 1;

            hull_point_tables_in[current_hull] = hull_point_table;
            hull_edge_tables_in[current_hull]  = hull_edge_table;
            hull_bone_tables_in[current_hull]  = hull_bone_table;
        }
    }
}

__kernel void egress_points(__global float4 *points_in,
                            __global int *point_vertex_references_in,
                            __global int *point_hull_indices_in,
                            __global short *point_hit_counts_in,
                            __global int *point_flags_in,
                            __global int4 *point_bone_tables_in,
                            __global float4 *points_out,
                            __global int *point_vertex_references_out,
                            __global int *point_hull_indices_out,
                            __global short *point_hit_counts_out,
                            __global int *point_flags_out,
                            __global int4 *point_bone_tables_out,
                            __global int *new_points,
                            int max_point)
{
    int current_point = get_global_id(0);
    if (current_point >= max_point) return;
    int new_point = new_points[current_point];
    if (new_point == -1) return;
    points_out[new_point]                  = points_in[current_point];
    point_vertex_references_out[new_point] = point_vertex_references_in[current_point];
    point_hull_indices_out[new_point]      = point_hull_indices_in[current_point];
    point_flags_out[new_point]             = point_flags_in[current_point];
    point_hit_counts_out[new_point]        = point_hit_counts_in[current_point];
    point_bone_tables_out[new_point]       = point_bone_tables_in[current_point];
}

__kernel void egress_edges(__global int2 *edges_in,
                           __global float *edge_lengths_in,
                           __global int *edge_flags_in,
                           __global int2 *edges_out,
                           __global float *edge_lengths_out,
                           __global int *edge_flags_out,
                           __global int *new_edges,
                           int max_edge)
{
    int current_edge = get_global_id(0);
    if (current_edge >= max_edge) return;
    int new_edge = new_edges[current_edge];
    if (new_edge == -1) return;
    edges_out[new_edge]        = edges_in[current_edge];
    edge_lengths_out[new_edge] = edge_lengths_in[current_edge];
    edge_flags_out[new_edge]   = edge_flags_in[current_edge];
}

__kernel void egress_hulls(__global float4 *hulls_in,
                           __global float2 *hull_scales_in,
                           __global float2 *hull_rotations_in,
                           __global float *hull_frictions_in,
                           __global float *hull_restitutions_in,
                           __global int2 *hull_point_tables_in,
                           __global int2 *hull_edge_tables_in,
                           __global int2 *hull_bone_tables_in,
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
                           __global int2 *hull_bone_tables_out,
                           __global int *hull_entity_ids_out,
                           __global int *hull_flags_out,
                           __global int *hull_mesh_ids_out,
                           __global int *hull_uv_offsets_out,
                           __global int *hull_integrity_out,
                           __global int *new_hulls,
                           int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int new_hull = new_hulls[current_hull];
    if (new_hull == -1) return;

    hulls_out[new_hull]             = hulls_in[current_hull];
    hull_scales_out[new_hull]       = hull_scales_in[current_hull];
    hull_rotations_out[new_hull]    = hull_rotations_in[current_hull];
    hull_frictions_out[new_hull]    = hull_frictions_in[current_hull];
    hull_restitutions_out[new_hull] = hull_restitutions_in[current_hull];
    hull_point_tables_out[new_hull] = hull_point_tables_in[current_hull];
    hull_edge_tables_out[new_hull]  = hull_edge_tables_in[current_hull];
    hull_bone_tables_out[new_hull]  = hull_bone_tables_in[current_hull];
    hull_entity_ids_out[new_hull]   = hull_entity_ids_in[current_hull];
    hull_flags_out[new_hull]        = hull_flags_in[current_hull];
    hull_mesh_ids_out[new_hull]     = hull_mesh_ids_in[current_hull];
    hull_uv_offsets_out[new_hull]   = hull_uv_offsets_in[current_hull];
    hull_integrity_out[new_hull]    = hull_integrity_in[current_hull];
}

__kernel void egress_hull_bones(__global float16 *hull_bones_in,
                                __global int *hull_bind_pose_indices_in,
                                __global int *hull_inv_bind_pose_indicies_in,
                                __global float16 *hull_bones_out,
                                __global int *hull_bind_pose_indicies_out,
                                __global int *hull_inv_bind_pose_indicies_out,
                                __global int *new_hull_bones,
                                int max_hull_bone)
{
    int current_hull_bone = get_global_id(0);
    if (current_hull_bone >= max_hull_bone) return;
    int new_hull_bone = new_hull_bones[current_hull_bone];
    if (new_hull_bone == -1) return;
    hull_bones_out[new_hull_bone]                  = hull_bones_in[current_hull_bone];
    hull_bind_pose_indicies_out[new_hull_bone]     = hull_bind_pose_indices_in[current_hull_bone];
    hull_inv_bind_pose_indicies_out[new_hull_bone] = hull_inv_bind_pose_indicies_in[current_hull_bone];
}

__kernel void egress_entity_bones(__global float16 *entity_bones_in,
                                  __global int *entity_bone_reference_ids_in,
                                  __global int *entity_bone_parent_ids_in,
                                  __global float16 *entity_bones_out,
                                  __global int *entity_bone_reference_ids_out,
                                  __global int *entity_bone_parent_ids_out,
                                  __global int *new_entity_bones,
                                  int max_entity_bone)
{
    int current_entity_bone = get_global_id(0);
    if (current_entity_bone >= max_entity_bone) return;
    int new_entity_bone = new_entity_bones[current_entity_bone];
    if (new_entity_bone == -1) return;    
    entity_bones_out[new_entity_bone]              = entity_bones_in[current_entity_bone];   ;
    entity_bone_reference_ids_out[new_entity_bone] = entity_bone_reference_ids_in[current_entity_bone];
    entity_bone_parent_ids_out[new_entity_bone]    = entity_bone_parent_ids_in[current_entity_bone];
}

__kernel void place_block(__global float4 *entities,
                          __global int2 *entity_hull_tables,
                          __global float4 *hulls,
                          __global int2 *hull_point_tables,
                          __global float2 *hull_rotations,
                          __global float4 *points,
                          int src,
                          int dest)
{
    entities[dest] = entities[src];
    
    int hull_id_src = entity_hull_tables[src].x;
    int hull_id_dest = entity_hull_tables[dest].x;

    hulls[hull_id_dest] = hulls[hull_id_src];
    hull_rotations[hull_id_dest] = hull_rotations[hull_id_src];
    
    int p0_id_src = hull_point_tables[hull_id_src].x;
    int p1_id_src = p0_id_src + 1;
    int p2_id_src = p0_id_src + 2;
    int p3_id_src = p0_id_src + 3;

    int p0_id_dest = hull_point_tables[hull_id_dest].x;
    int p1_id_dest = p0_id_dest + 1;
    int p2_id_dest = p0_id_dest + 2;
    int p3_id_dest = p0_id_dest + 3;

    points[p0_id_dest] = points[p0_id_src];
    points[p1_id_dest] = points[p1_id_src];
    points[p2_id_dest] = points[p2_id_src];
    points[p3_id_dest] = points[p3_id_src];
}


// todo: implement for armature
__kernel void rotate_hull(__global float4 *hulls,
                          __global int2 *hull_point_tables,
                          __global float4 *points,
                          int target,
                          float angle)
{
    float4 hull        = hulls[target];
    int2 point_table   = hull_point_tables[target];
    int start          = point_table.x;
    int end            = point_table.y;
    float2 origin      = (float2)(hull.x, hull.y);
    for (int i = start; i <= end; i++)
    {
        float4 point = points[i];
        points[i] = rotate_point(point, origin, angle);
    }
}
