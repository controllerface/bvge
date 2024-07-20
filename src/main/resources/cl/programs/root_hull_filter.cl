/*
    Filters all instances of a specific model in one array.
*/
__kernel void root_hull_filter(__global int *entity_root_hulls,
                               __global int *entity_model_indices,
                               __global int *hulls_out,
                               __global int *counter,
                               int target_model_id,
                               int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    int root_hull = entity_root_hulls[current_entity];
    int model_id = entity_model_indices[current_entity];
    if(model_id == target_model_id)
    {
        int i = atomic_inc(&counter[0]);
        hulls_out[i] = root_hull;
    }
}


/*
    Get the count of specific hull in array.
*/
__kernel void root_hull_count(__global int *entity_model_indices,
                              __global int *counter,
                              int target_model_id,
                              int max_entity)
{
    int current_entity = get_global_id(0);
    if (current_entity >= max_entity) return;
    int model_id = entity_model_indices[current_entity];
    if(model_id == target_model_id)
    {
        atomic_inc(&counter[0]);
    }
}




__kernel void hull_filter(__global int *hull_mesh_ids,
                          __global int *hulls_out,
                          __global int *counter,
                          int target_mesh_id,
                          int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int mesh_id = hull_mesh_ids[current_hull];
    if(mesh_id == target_mesh_id)
    {
        int i = atomic_inc(&counter[0]);
        hulls_out[i] = current_hull;
    }
}


__kernel void hull_count(__global int *hull_mesh_ids,
                         __global int *counter,
                         int target_mesh_id,
                         int max_hull)
{
    int current_hull = get_global_id(0);
    if (current_hull >= max_hull) return;
    int mesh_id = hull_mesh_ids[current_hull];
    if(mesh_id == target_mesh_id)
    {
        atomic_inc(&counter[0]);
    }
}

