package com.controllerface.bvge.memory.sectors;

import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.crud.*;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.crud.GPUCrud;
import com.controllerface.bvge.memory.GPUCoreMemory;
import com.controllerface.bvge.memory.SectorContainer;
import com.controllerface.bvge.memory.groups.CoreBufferGroup;

import static com.controllerface.bvge.gpu.GPU.CL.arg_long;

public class OrderedSectorInput implements SectorContainer, GPUResource
{
    private static final long ENTITY_INIT = 1_000L;
    private static final long HULL_INIT   = 1_000L;
    private static final long EDGE_INIT   = 2_400L;
    private static final long POINT_INIT  = 5_000L;

    private final GPUProgram p_gpu_crud;

    private final GPUKernel k_merge_point;
    private final GPUKernel k_merge_edge;
    private final GPUKernel k_merge_hull;
    private final GPUKernel k_merge_entity;
    private final GPUKernel k_merge_hull_bone;
    private final GPUKernel k_merge_entity_bone;

    private final CoreBufferGroup buffers;
    private final SectorController controller;

    public OrderedSectorInput(CL_CommandQueue cmd_queue, GPUCoreMemory core_memory)
    {
        this.p_gpu_crud = new GPUCrud().init();
        this.buffers    = new CoreBufferGroup(cmd_queue, "Sector Ingress", ENTITY_INIT, HULL_INIT, EDGE_INIT, POINT_INIT);
        this.controller = new SectorController(cmd_queue, this.p_gpu_crud, this.buffers);

        k_merge_point       = new MergePoint_k(cmd_queue, p_gpu_crud).init(core_memory, buffers);
        k_merge_edge        = new MergeEdge_k(cmd_queue, p_gpu_crud).init(core_memory, buffers);
        k_merge_hull        = new MergeHull_k(cmd_queue, p_gpu_crud).init(core_memory, buffers);
        k_merge_entity      = new MergeEntity_k(cmd_queue, p_gpu_crud).init(core_memory, buffers);
        k_merge_hull_bone   = new MergeHullBone_k(cmd_queue, p_gpu_crud).init(core_memory, buffers);
        k_merge_entity_bone = new MergeEntityBone_k(cmd_queue, p_gpu_crud).init(core_memory, buffers);
    }

    public void merge_into(SectorContainer target_container)
    {
        int point_count       = controller.next_point();
        int edge_count        = controller.next_edge();
        int hull_count        = controller.next_hull();
        int entity_count      = controller.next_entity();
        int hull_bone_count   = controller.next_hull_bone();
        int entity_bone_count = controller.next_entity_bone();

        int point_size       = GPU.compute.calculate_preferred_global_size(point_count);
        int edge_size        = GPU.compute.calculate_preferred_global_size(edge_count);
        int hull_size        = GPU.compute.calculate_preferred_global_size(hull_count);
        int entity_size      = GPU.compute.calculate_preferred_global_size(entity_count);
        int hull_bone_size   = GPU.compute.calculate_preferred_global_size(hull_bone_count);
        int entity_bone_size = GPU.compute.calculate_preferred_global_size(entity_bone_count);

        if (point_count > 0) k_merge_point
            .set_arg(MergePoint_k.Args.point_offset, target_container.next_point())
            .set_arg(MergePoint_k.Args.bone_offset,  target_container.next_hull_bone())
            .set_arg(MergePoint_k.Args.hull_offset,  target_container.next_hull())
            .set_arg(MergePoint_k.Args.max_point,    point_count)
            .call(arg_long(point_size), GPU.compute.preferred_work_size);

        if (edge_count > 0) k_merge_edge
            .set_arg(MergeEdge_k.Args.edge_offset,  target_container.next_edge())
            .set_arg(MergeEdge_k.Args.point_offset, target_container.next_point())
            .set_arg(MergeEdge_k.Args.max_edge,     edge_count)
            .call(arg_long(edge_size), GPU.compute.preferred_work_size);

        if (hull_count > 0) k_merge_hull
            .set_arg(MergeHull_k.Args.hull_offset,      target_container.next_hull())
            .set_arg(MergeHull_k.Args.point_offset,     target_container.next_point())
            .set_arg(MergeHull_k.Args.edge_offset,      target_container.next_edge())
            .set_arg(MergeHull_k.Args.entity_offset,    target_container.next_entity())
            .set_arg(MergeHull_k.Args.hull_bone_offset, target_container.next_hull_bone())
            .set_arg(MergeHull_k.Args.max_hull,         hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        if (entity_count > 0) k_merge_entity
            .set_arg(MergeEntity_k.Args.entity_offset,        target_container.next_entity())
            .set_arg(MergeEntity_k.Args.hull_offset,          target_container.next_hull())
            .set_arg(MergeEntity_k.Args.armature_bone_offset, target_container.next_entity_bone())
            .set_arg(MergeEntity_k.Args.max_entity,           edge_count)
            .call(arg_long(entity_size), GPU.compute.preferred_work_size);

        if (hull_bone_count > 0) k_merge_hull_bone
            .set_arg(MergeHullBone_k.Args.hull_bone_offset,     target_container.next_hull_bone())
            .set_arg(MergeHullBone_k.Args.armature_bone_offset, target_container.next_entity_bone())
            .set_arg(MergeHullBone_k.Args.max_hull_bone,        hull_bone_count)
            .call(arg_long(hull_bone_size), GPU.compute.preferred_work_size);

        if (entity_bone_count > 0) k_merge_entity_bone
            .set_arg(MergeEntityBone_k.Args.armature_bone_offset, target_container.next_entity_bone())
            .set_arg(MergeEntityBone_k.Args.max_entity_bone,      entity_bone_count)
            .call(arg_long(entity_bone_size), GPU.compute.preferred_work_size);

        controller.reset();
    }

    @Override
    public int next_point()
    {
        return controller.next_point();
    }

    @Override
    public int next_edge()
    {
        return controller.next_edge();
    }

    @Override
    public int next_hull()
    {
        return controller.next_hull();
    }

    @Override
    public int next_entity()
    {
        return controller.next_entity();
    }

    @Override
    public int next_hull_bone()
    {
        return controller.next_hull_bone();
    }

    @Override
    public int next_entity_bone()
    {
        return controller.next_entity_bone();
    }

    @Override
    public int create_point(float[] position, int[] bone_ids, int vertex_index, int hull_index, int hit_count, int flags)
    {
        return controller.create_point(position, bone_ids, vertex_index, hull_index, hit_count, flags);
    }

    @Override
    public int create_edge(int p1, int p2, float l, int flags, int edge_pin)
    {
        return controller.create_edge(p1, p2, l, flags, edge_pin);
    }

    @Override
    public int create_hull(int mesh_id, float[] position, float[] scale, float[] rotation, int[] point_table, int[] edge_table, int[] bone_table, float friction, float restitution, int entity_id, int uv_offset, int flags)
    {
        return controller.create_hull(mesh_id, position, scale, rotation, point_table, edge_table, bone_table, friction, restitution, entity_id, uv_offset, flags);
    }

    @Override
    public int create_entity(float x, float y, float z, float w, int[] hull_table, int[] bone_table, float mass, int anim_index, float anim_time, int root_hull, int model_id, int model_transform_id, int type, int flags)
    {
        return controller.create_entity(x, y, z, w, hull_table, bone_table, mass, anim_index, anim_time, root_hull, model_id, model_transform_id, type, flags);
    }

    @Override
    public int create_hull_bone(float[] bone_data, int bind_pose_id, int inv_bind_pose_id)
    {
        return controller.create_hull_bone(bone_data, bind_pose_id, inv_bind_pose_id);
    }

    @Override
    public int create_entity_bone(int bone_reference, int bone_parent_id, float[] bone_data)
    {
        return controller.create_entity_bone(bone_reference, bone_parent_id, bone_data);
    }

    @Override
    public void release()
    {
        p_gpu_crud.release();
        buffers.release();
        controller.release();
    }
}
