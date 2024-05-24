/*
    Filters all instances of a specific model in one array.
*/
__kernel void root_hull_filter(__global int *entity_root_hulls,
                               __global int *entity_model_indices,
                               __global int *hulls_out,
                               __global int *counter,
                               int target_model_id)
{
    int current_entity = get_global_id(0);
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
                              int target_model_id)
{
    int current_entity = get_global_id(0);
    int model_id = entity_model_indices[current_entity];
    if(model_id == target_model_id)
    {
        atomic_inc(&counter[0]);
    }
}
