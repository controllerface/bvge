/*
    Filters all instances of a specific model in one array.
*/
__kernel void root_hull_filter( __global int4 *armature_flags,
                                __global int *hulls_out,
                                __global int *counter,
                                int model_id)
{
    int gid = get_global_id(0);
    int4 flag_data = armature_flags[gid]; // x root hull   y = model id

    if(flag_data.y == model_id){
        int i = atomic_inc(&counter[0]);
        hulls_out[i] = flag_data.x;
    }
}


/*
    Get the count of specific hull in array.
*/
__kernel void root_hull_count( __global int4 *armature_flags,
                                __global int *counter,
                                int model_id)
{
    int gid = get_global_id(0);
    int4 flag_data = armature_flags[gid]; // x root hull   y = model id

    if(flag_data.y == model_id){
        atomic_inc(&counter[0]);
    }
}