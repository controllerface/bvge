package com.controllerface.bvge.cl.kernels;

public enum Kernel
{
    aabb_collide,
    animate_entities,
    animate_bones,
    animate_points,
    apply_reactions,
    build_key_map,
    calculate_batch_offsets,
    compact_entity_bones,
    compact_entities,
    compact_hull_bones,
    compact_edges,
    compact_hulls,
    compact_points,
    complete_bounds_multi_block,
    complete_candidates_multi_block_out,
    complete_deletes_multi_block_out,
    complete_int2_multi_block,
    complete_int4_multi_block,
    complete_int_multi_block,
    complete_int_multi_block_out,
    count_candidates,
    count_egress_entities,
    count_mesh_batches,
    count_mesh_instances,
    create_animation_timings,
    create_entity,
    create_entity_bone,
    create_hull_bone,
    create_bone_bind_pose,
    create_bone_channel,
    create_bone_reference,
    create_edge,
    create_hull,
    create_keyframe,
    create_mesh_face,
    create_mesh_reference,
    create_model_transform,
    create_point,
    create_texture_uv,
    create_vertex_reference,
    egress_entities,
    egress_broken,
    egress_collected,
    finalize_candidates,
    generate_keys,
    integrate,
    integrate_entities,
    locate_in_bounds,
    merge_point,
    merge_edge,
    merge_hull,
    merge_entity,
    merge_hull_bone,
    merge_entity_bone,
    move_entities,
    move_hulls,
    prepare_entities,
    prepare_bones,
    prepare_bounds,
    prepare_edges,
    prepare_points,
    prepare_liquids,
    prepare_transforms,
    read_position,
    resolve_constraints,
    root_hull_count,
    root_hull_filter,
    rotate_hull,
    sat_collide,
    scan_bounds_multi_block,
    scan_bounds_single_block,
    scan_candidates_multi_block_out,
    scan_candidates_single_block_out,
    scan_deletes_multi_block_out,
    scan_deletes_single_block_out,
    scan_int2_multi_block,
    scan_int2_single_block,
    scan_int4_multi_block,
    scan_int4_single_block,
    scan_int_multi_block,
    scan_int_multi_block_out,
    scan_int_single_block,
    scan_int_single_block_out,
    set_bone_channel_table,
    sort_reactions,
    transfer_detail_data,
    transfer_render_data,
    update_accel,
    update_mouse_position,
    write_mesh_details,

    update_select_block,
    clear_select_block,
    place_block,

    set_control_points,
    handle_movement,
}
