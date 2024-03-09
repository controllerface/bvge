package com.controllerface.bvge.gpu;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.kernels.*;
import com.controllerface.bvge.cl.programs.GpuCrud;

import static com.controllerface.bvge.cl.CLUtils.arg_float2;
import static com.controllerface.bvge.cl.CLUtils.arg_float4;

/**
 * The "Reference Memory" interface
 * -
 * Calling code uses this object as a single access point to the core memory that
 * is involved in the game engine's operation, specifically for data that is to be
 * considered "reference data" for example, the vertices and faces that make up a
 * model, the bone information for animations, etc. The memory being used is resident
 * on the GPU, so this class is essentially a direct interface into GPU memory.
 * This class keeps track of the number of "top-level" memory objects, which are
 * generally defined as any "logical object" that is composed of various primitive
 * values.
 * Typically, a method is exposed, that accepts as parameters, the various
 * components that describe an object. These methods in turn delegate the memory
 * access calls to the GPU APIs. In practice, this means that the actual buffers
 * that store the components of created objects may reside in separate memory
 * regions depending on the layout and type of the data being stored.
 * For example, consider an object that has two float components and one integer
 * component, a method to create this object may have a signature like this:
 * -
 *    foo(float x, float y, int i)
 * -
 * The GPU implementation may store all three of these values in one continuous
 * block of memory, or it may store the x and y components together, and the int
 * value in a separate memory segment, or even store all three in completely
 * different sections of memory. As such, this class cannot make too many
 * assumptions about the memory layout of all the various components. However,
 * each top-level object does have one known "memory width" that is used to keep
 * track of the number of objects of that type that are currently stored in
 * memory.
 * For these base objects, they will always be laid out in a continuous manner.
 * Just note that this width generally applies to one or a small number of "base"
 * properties of the object, and any extra properties or meta-data related to the
 * object will use separate, but index-aligned, memory space. The best existing
 * example of this concept are the HULL object type. The "main" value tracked for
 * these objects is represented on the GPU as a float4, with the first two values
 * (x,y) designating the world-space position of the hull center, and the second
 * two values (z,w) defining the width and height scaling of the hull. This base
 * set of values has a width of 4, so the hull index increments by 4 when each new
 * hull is created. Hulls however also have other data, for example an indexing
 * table that defines the start/end indices in the point and edge buffers of the
 * points and edges that are part of the hull. These values are stored in buffers
 * that align with the hull buffer, so that an index into one buffer can be used
 * interchangeably with the buffers that store all the other components of the
 * object, making access to a single "object" possible by indexing into all the
 * aligned arrays with the same index.
 */
public class GPUReferenceMemory
{
    private int hull_index            = 0;
    private int point_index           = 0;
    private int edge_index            = 0;
    private int vertex_ref_index      = 0;
    private int bone_bind_index       = 0;
    private int bone_ref_index        = 0;
    private int bone_index            = 0;
    private int model_transform_index = 0;
    private int armature_bone_index   = 0;
    private int armature_index        = 0;
    private int mesh_index            = 0;
    private int face_index            = 0;
    private int uv_index              = 0;
    private int keyframe_index        = 0;
    private int bone_channel_index    = 0;
    private int animation_index       = 0;

    private final GPUProgram gpu_crud = new GpuCrud();

    private GPUKernel create_animation_timings_k;
    private GPUKernel create_armature_k;
    private GPUKernel create_armature_bone_k;
    private GPUKernel create_bone_k;
    private GPUKernel create_bone_bind_pose_k;
    private GPUKernel create_bone_channel_k;
    private GPUKernel create_bone_reference_k;
    private GPUKernel create_edge_k;
    private GPUKernel create_hull_k;
    private GPUKernel create_keyframe_k;
    private GPUKernel create_mesh_face_k;
    private GPUKernel create_mesh_reference_k;
    private GPUKernel create_model_transform_k;
    private GPUKernel create_point_k;
    private GPUKernel create_texture_uv_k;
    private GPUKernel create_vertex_reference_k;

    public GPUReferenceMemory()
    {
        init();
    }

    private void init()
    {
        gpu_crud.init();

        long create_point_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_point);
        create_point_k = new CreatePoint_k(GPU.command_queue_ptr, create_point_k_ptr)
            .mem_arg(CreatePoint_k.Args.points, GPU.Buffer.points.memory)
            .mem_arg(CreatePoint_k.Args.vertex_tables, GPU.Buffer.point_vertex_tables.memory)
            .mem_arg(CreatePoint_k.Args.bone_tables, GPU.Buffer.point_bone_tables.memory);

        long create_texture_uv_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_texture_uv);
        create_texture_uv_k = new CreateTextureUV_k(GPU.command_queue_ptr, create_texture_uv_ptr)
            .mem_arg(CreateTextureUV_k.Args.texture_uvs, GPU.Buffer.texture_uvs.memory);

        long create_edge_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_edge);
        create_edge_k = new CreateEdge_k(GPU.command_queue_ptr, create_edge_k_ptr)
            .mem_arg(CreateEdge_k.Args.edges, GPU.Buffer.edges.memory);

        long create_keyframe_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_keyframe);
        create_keyframe_k = new CreateKeyFrame_k(GPU.command_queue_ptr, create_keyframe_k_ptr)
            .mem_arg(CreateKeyFrame_k.Args.key_frames, GPU.Buffer.key_frames.memory)
            .mem_arg(CreateKeyFrame_k.Args.frame_times, GPU.Buffer.frame_times.memory);

        long create_vertex_reference_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_vertex_reference);
        create_vertex_reference_k = new CreateVertexRef_k(GPU.command_queue_ptr, create_vertex_reference_k_ptr)
            .mem_arg(CreateVertexRef_k.Args.vertex_references, GPU.Buffer.vertex_references.memory)
            .mem_arg(CreateVertexRef_k.Args.vertex_weights, GPU.Buffer.vertex_weights.memory)
            .mem_arg(CreateVertexRef_k.Args.uv_tables, GPU.Buffer.uv_tables.memory);

        long create_bone_bind_pose_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone_bind_pose);
        create_bone_bind_pose_k = new CreateBoneBindPose_k(GPU.command_queue_ptr, create_bone_bind_pose_k_ptr)
            .mem_arg(CreateBoneBindPose_k.Args.bone_bind_poses, GPU.Buffer.bone_bind_poses.memory)
            .mem_arg(CreateBoneBindPose_k.Args.bone_bind_parents, GPU.Buffer.bone_bind_parents.memory);

        long create_bone_reference_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone_reference);
        create_bone_reference_k = new CreateBoneRef_k(GPU.command_queue_ptr, create_bone_reference_k_ptr)
            .mem_arg(CreateBoneRef_k.Args.bone_references, GPU.Buffer.bone_references.memory);

        long create_bone_channel_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone_channel);
        create_bone_channel_k = new CreateBoneChannel_k(GPU.command_queue_ptr, create_bone_channel_k_ptr)
            .mem_arg(CreateBoneChannel_k.Args.animation_timing_indices, GPU.Buffer.animation_timing_indices.memory)
            .mem_arg(CreateBoneChannel_k.Args.bone_pos_channel_tables, GPU.Buffer.bone_pos_channel_tables.memory)
            .mem_arg(CreateBoneChannel_k.Args.bone_rot_channel_tables, GPU.Buffer.bone_rot_channel_tables.memory)
            .mem_arg(CreateBoneChannel_k.Args.bone_scl_channel_tables, GPU.Buffer.bone_scl_channel_tables.memory);

        long create_armature_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_armature);
        create_armature_k = new CreateArmature_k(GPU.command_queue_ptr, create_armature_k_ptr)
            .mem_arg(CreateArmature_k.Args.armatures, GPU.Buffer.armatures.memory)
            .mem_arg(CreateArmature_k.Args.armature_flags, GPU.Buffer.armature_flags.memory)
            .mem_arg(CreateArmature_k.Args.hull_tables, GPU.Buffer.armature_hull_table.memory)
            .mem_arg(CreateArmature_k.Args.armature_masses, GPU.Buffer.armature_mass.memory)
            .mem_arg(CreateArmature_k.Args.armature_animation_indices, GPU.Buffer.armature_animation_indices.memory)
            .mem_arg(CreateArmature_k.Args.armature_animation_elapsed, GPU.Buffer.armature_animation_elapsed.memory);

        long create_bone_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_bone);
        create_bone_k = new CreateBone_k(GPU.command_queue_ptr, create_bone_k_ptr)
            .mem_arg(CreateBone_k.Args.bones, GPU.Buffer.bone_instances.memory)
            .mem_arg(CreateBone_k.Args.bone_index_tables, GPU.Buffer.bone_index_tables.memory);

        long create_armature_bone_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_armature_bone);
        create_armature_bone_k = new CreateArmatureBone_k(GPU.command_queue_ptr, create_armature_bone_k_ptr)
            .mem_arg(CreateArmatureBone_k.Args.armature_bones, GPU.Buffer.armatures_bones.memory)
            .mem_arg(CreateArmatureBone_k.Args.bone_bind_tables, GPU.Buffer.bone_bind_tables.memory);

        long create_model_transform_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_model_transform);
        create_model_transform_k = new CreateModelTransform_k(GPU.command_queue_ptr, create_model_transform_k_ptr)
            .mem_arg(CreateModelTransform_k.Args.model_transforms, GPU.Buffer.model_transforms.memory);

        long create_hull_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_hull);
        create_hull_k = new CreateHull_k(GPU.command_queue_ptr, create_hull_k_ptr)
            .mem_arg(CreateHull_k.Args.hulls, GPU.Buffer.hulls.memory)
            .mem_arg(CreateHull_k.Args.hull_rotations, GPU.Buffer.hull_rotation.memory)
            .mem_arg(CreateHull_k.Args.element_tables, GPU.Buffer.hull_element_tables.memory)
            .mem_arg(CreateHull_k.Args.hull_flags, GPU.Buffer.hull_flags.memory)
            .mem_arg(CreateHull_k.Args.hull_mesh_ids, GPU.Buffer.hull_mesh_ids.memory);

        long create_mesh_reference_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_mesh_reference);
        create_mesh_reference_k = new CreateMeshReference_k(GPU.command_queue_ptr, create_mesh_reference_k_ptr)
            .mem_arg(CreateMeshReference_k.Args.mesh_ref_tables, GPU.Buffer.mesh_references.memory);

        long create_mesh_face_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_mesh_face);
        create_mesh_face_k = new CreateMeshFace_k(GPU.command_queue_ptr, create_mesh_face_k_ptr)
            .mem_arg(CreateMeshFace_k.Args.mesh_faces, GPU.Buffer.mesh_faces.memory);

        long create_animation_timings_k_ptr = gpu_crud.kernel_ptr(GPU.Kernel.create_animation_timings);
        create_animation_timings_k = new CreateAnimationTimings_k(GPU.command_queue_ptr, create_animation_timings_k_ptr)
            .mem_arg(CreateAnimationTimings_k.Args.animation_timings, GPU.Buffer.animation_timings.memory);
    }

    // index methods

    public int next_mesh()
    {
        return mesh_index;
    }

    public int next_armature()
    {
        return armature_index;
    }

    public int next_hull()
    {
        return hull_index;
    }

    public int next_point()
    {
        return point_index;
    }

    public int next_edge()
    {
        return edge_index;
    }

    public int next_bone()
    {
        return bone_index;
    }

    public int next_armature_bone()
    {
        return armature_bone_index;
    }


    // creation methods

    public int new_animation_timings(double[] timings)
    {
        create_animation_timings_k
            .set_arg(CreateAnimationTimings_k.Args.target, animation_index)
            .set_arg(CreateAnimationTimings_k.Args.new_animation_timing, timings)
            .call(GPU.global_single_size);

        return animation_index++;
    }

    public int new_bone_channel(int anim_timing_index, int[] pos_table, int[] rot_table, int[] scl_table)
    {
        create_bone_channel_k
            .set_arg(CreateBoneChannel_k.Args.target, bone_channel_index)
            .set_arg(CreateBoneChannel_k.Args.new_animation_timing_index, anim_timing_index)
            .set_arg(CreateBoneChannel_k.Args.new_bone_pos_channel_table, pos_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_rot_channel_table, rot_table)
            .set_arg(CreateBoneChannel_k.Args.new_bone_scl_channel_table, scl_table)
            .call(GPU.global_single_size);

        return bone_channel_index++;
    }

    public int new_keyframe(float[] frame, double time)
    {
        create_keyframe_k
            .set_arg(CreateKeyFrame_k.Args.target, keyframe_index)
            .set_arg(CreateKeyFrame_k.Args.new_keyframe, frame)
            .set_arg(CreateKeyFrame_k.Args.new_frame_time, time)
            .call(GPU.global_single_size);

        return keyframe_index++;
    }

    public int new_texture_uv(float u, float v)
    {
        create_texture_uv_k
            .set_arg(CreateTextureUV_k.Args.target, uv_index)
            .set_arg(CreateTextureUV_k.Args.new_texture_uv, arg_float2(u, v))
            .call(GPU.global_single_size);

        return uv_index++;
    }

    public int new_edge(int p1, int p2, float l, int flags)
    {
        create_edge_k
            .set_arg(CreateEdge_k.Args.target, edge_index)
            .set_arg(CreateEdge_k.Args.new_edge, arg_float4(p1, p2, l, flags))
            .call(GPU.global_single_size);

        return edge_index++;
    }

    public int new_point(float[] position, int[] vertex_table, int[] bone_ids)
    {
        var new_point = new float[]{position[0], position[1], position[0], position[1]};
        create_point_k
            .set_arg(CreatePoint_k.Args.target, point_index)
            .set_arg(CreatePoint_k.Args.new_point, new_point)
            .set_arg(CreatePoint_k.Args.new_vertex_table, vertex_table)
            .set_arg(CreatePoint_k.Args.new_bone_table, bone_ids)
            .call(GPU.global_single_size);

        return point_index++;
    }

    public int new_hull(int mesh_id, float[] transform, float[] rotation, int[] table, int[] flags)
    {
        create_hull_k
            .set_arg(CreateHull_k.Args.target, hull_index)
            .set_arg(CreateHull_k.Args.new_hull, transform)
            .set_arg(CreateHull_k.Args.new_rotation, rotation)
            .set_arg(CreateHull_k.Args.new_table, table)
            .set_arg(CreateHull_k.Args.new_flags, flags)
            .set_arg(CreateHull_k.Args.new_hull_mesh_id, mesh_id)
            .call(GPU.global_single_size);

        return hull_index++;
    }

    public int new_mesh_reference(int[] mesh_ref_table)
    {
        create_mesh_reference_k
            .set_arg(CreateMeshReference_k.Args.target, mesh_index)
            .set_arg(CreateMeshReference_k.Args.new_mesh_ref_table, mesh_ref_table)
            .call(GPU.global_single_size);

        return mesh_index++;
    }

    public int new_mesh_face(int[] face)
    {
        create_mesh_face_k
            .set_arg(CreateMeshFace_k.Args.target, face_index)
            .set_arg(CreateMeshFace_k.Args.new_mesh_face, face)
            .call(GPU.global_single_size);

        return face_index++;
    }

    public int new_armature(float x, float y, int[] table, int[] flags, float mass, int anim_index, double anim_time)
    {
        create_armature_k
            .set_arg(CreateArmature_k.Args.target, armature_index)
            .set_arg(CreateArmature_k.Args.new_armature, arg_float4(x, y, x, y))
            .set_arg(CreateArmature_k.Args.new_armature_flags, flags)
            .set_arg(CreateArmature_k.Args.new_hull_table, table)
            .set_arg(CreateArmature_k.Args.new_armature_mass, mass)
            .set_arg(CreateArmature_k.Args.new_armature_animation_index, anim_index)
            .set_arg(CreateArmature_k.Args.new_armature_animation_time, anim_time)
            .call(GPU.global_single_size);

        return armature_index++;
    }

    public int new_vertex_reference(float x, float y, float[] weights, int[] uv_table)
    {
        create_vertex_reference_k
            .set_arg(CreateVertexRef_k.Args.target, vertex_ref_index)
            .set_arg(CreateVertexRef_k.Args.new_vertex_reference, arg_float2(x, y))
            .set_arg(CreateVertexRef_k.Args.new_vertex_weights, weights)
            .set_arg(CreateVertexRef_k.Args.new_uv_table, uv_table)
            .call(GPU.global_single_size);

        return vertex_ref_index++;
    }

    public int new_bone_bind_pose(int bind_parent, float[] bone_data)
    {
        create_bone_bind_pose_k
            .set_arg(CreateBoneBindPose_k.Args.target,bone_bind_index)
            .set_arg(CreateBoneBindPose_k.Args.new_bone_bind_pose, bone_data)
            .set_arg(CreateBoneBindPose_k.Args.bone_bind_parent, bind_parent)
            .call(GPU.global_single_size);

        return bone_bind_index++;
    }

    public int new_bone_reference(float[] bone_data)
    {
        create_bone_reference_k
            .set_arg(CreateBoneRef_k.Args.target, bone_ref_index)
            .set_arg(CreateBoneRef_k.Args.new_bone_reference, bone_data)
            .call(GPU.global_single_size);

        return bone_ref_index++;
    }

    public int new_bone(int[] bone_table, float[] bone_data)
    {
        create_bone_k
            .set_arg(CreateBone_k.Args.target, bone_index)
            .set_arg(CreateBone_k.Args.new_bone, bone_data)
            .set_arg(CreateBone_k.Args.new_bone_table, bone_table)
            .call(GPU.global_single_size);

        return bone_index++;
    }

    public int new_armature_bone(int[] bone_bind_table, float[] bone_data)
    {
        create_armature_bone_k
            .set_arg(CreateArmatureBone_k.Args.target, armature_bone_index)
            .set_arg(CreateArmatureBone_k.Args.new_armature_bone, bone_data)
            .set_arg(CreateArmatureBone_k.Args.new_bone_bind_table, bone_bind_table)
            .call(GPU.global_single_size);

        return armature_bone_index++;
    }

    public int new_model_transform(float[] transform_data)
    {
        create_model_transform_k
            .set_arg(CreateModelTransform_k.Args.target, model_transform_index)
            .set_arg(CreateModelTransform_k.Args.new_model_transform, transform_data)
            .call(GPU.global_single_size);

        return model_transform_index++;
    }

    public void compact_buffers(int edge_shift,
                                       int bone_shift,
                                       int point_shift,
                                       int hull_shift,
                                       int armature_shift,
                                       int armature_bone_shift)
    {
        edge_index          -= (edge_shift);
        bone_index          -= (bone_shift);
        point_index         -= (point_shift);
        hull_index          -= (hull_shift);
        armature_index      -= (armature_shift);
        armature_bone_index -= (armature_bone_shift);
    }
}
