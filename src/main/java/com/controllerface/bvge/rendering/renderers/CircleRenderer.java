package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.rendering.HullCount_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.HullFilter_k;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareTransforms_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PrepareTransforms;
import com.controllerface.bvge.gpu.cl.programs.rendering.RootHullFilter;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;
import com.controllerface.bvge.models.geometry.MeshRegistry;
import com.controllerface.bvge.rendering.HullIndexData;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;

public class CircleRenderer extends GameSystem
{
    public static final int CIRCLES_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;

    private static final int TRANSFORM_ATTRIBUTE = 0;

    private final GPUProgram p_prepare_transforms = new PrepareTransforms();
    private final GPUProgram p_root_hull_filter = new RootHullFilter();
    private GPUKernel k_prepare_transforms;
    private GPUKernel k_root_hull_filter;
    private GPUKernel k_root_hull_count;
    private GL_Shader shader;

    private GL_VertexArray vao;
    private GL_VertexBuffer vbo_transform;
    private long ptr_vbo_transform;
    private long svm_atomic_counter;

    private HullIndexData circle_hulls;

    public CircleRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        shader = GPU.GL.new_shader("circle_shader.glsl", GL_ShaderType.THREE_STAGE);
        vao = GPU.GL.new_vao();
        vbo_transform = GPU.GL.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, CIRCLES_BUFFER_SIZE);
        vao.enable_attribute(TRANSFORM_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_transforms.init();
        p_root_hull_filter.init();
        ptr_vbo_transform = GPGPU.share_memory(vbo_transform.id());
        svm_atomic_counter = GPGPU.cl_new_pinned_int();

        long k_ptr_prepare_transforms = p_prepare_transforms.kernel_ptr(KernelType.prepare_transforms);
        k_prepare_transforms = (new PrepareTransforms_k(GPGPU.compute.render_queue.ptr(), k_ptr_prepare_transforms))
            .ptr_arg(PrepareTransforms_k.Args.transforms_out, ptr_vbo_transform)
            .buf_arg(PrepareTransforms_k.Args.hull_positions, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL))
            .buf_arg(PrepareTransforms_k.Args.hull_scales, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_SCALE))
            .buf_arg(PrepareTransforms_k.Args.hull_rotations, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_ROTATION));

        long k_ptr_root_hull_filter = p_root_hull_filter.kernel_ptr(KernelType.hull_filter);
        k_root_hull_filter = new HullFilter_k(GPGPU.compute.render_queue.ptr(), k_ptr_root_hull_filter)
            .buf_arg(HullFilter_k.Args.hull_mesh_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_MESH_ID));

        long k_ptr_root_hull_count =  p_root_hull_filter.kernel_ptr(KernelType.hull_count);
        k_root_hull_count = new HullCount_k(GPGPU.compute.render_queue.ptr(), k_ptr_root_hull_count)
            .buf_arg(HullCount_k.Args.hull_mesh_ids, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_MESH_ID));
    }

    public HullIndexData hull_filter(long queue_ptr, int mesh_id)
    {
        GPGPU.cl_zero_buffer(queue_ptr, svm_atomic_counter, CL_DataTypes.cl_int.size());

        int hull_count = GPGPU.core_memory.sector_container().next_hull();
        int hull_size  = GPGPU.compute.calculate_preferred_global_size(hull_count);

        k_root_hull_count
            .ptr_arg(HullCount_k.Args.counter, svm_atomic_counter)
            .set_arg(HullCount_k.Args.mesh_id, mesh_id)
            .set_arg(HullCount_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPGPU.compute.preferred_work_size);

        int final_count = GPGPU.cl_read_pinned_int(queue_ptr, svm_atomic_counter);

        if (final_count == 0)
        {
            return new HullIndexData(-1, final_count);
        }

        long final_buffer_size = (long) CL_DataTypes.cl_int.size() * final_count;
        var hulls_out = GPGPU.cl_new_buffer(final_buffer_size);

        GPGPU.cl_zero_buffer(queue_ptr, svm_atomic_counter, CL_DataTypes.cl_int.size());

        k_root_hull_filter
            .ptr_arg(HullFilter_k.Args.hulls_out, hulls_out)
            .ptr_arg(HullFilter_k.Args.counter, svm_atomic_counter)
            .set_arg(HullFilter_k.Args.mesh_id, mesh_id)
            .set_arg(HullFilter_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPGPU.compute.preferred_work_size);

        return new HullIndexData(hulls_out, final_count);
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != -1)
        {
            GPGPU.cl_release_buffer(circle_hulls.indices());
        }

        circle_hulls = hull_filter(GPGPU.compute.render_queue.ptr(), MeshRegistry.CIRCLE_MESH);

        if (circle_hulls.count() == 0) return;

        vao.bind();

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPGPU.compute.calculate_preferred_global_size(count);
            k_prepare_transforms
                .share_mem(ptr_vbo_transform)
                .ptr_arg(PrepareTransforms_k.Args.indices, circle_hulls.indices())
                .set_arg(PrepareTransforms_k.Args.offset, offset)
                .set_arg(PrepareTransforms_k.Args.max_hull, count)
                .call(arg_long(count_size), GPGPU.compute.preferred_work_size);

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        shader.detach();
        vao.unbind();
    }

    @Override
    public void shutdown()
    {
        vao.release();
        vbo_transform.release();
        shader.release();
        p_prepare_transforms.release();
        p_root_hull_filter.release();
        GPGPU.cl_release_buffer(ptr_vbo_transform);
        GPGPU.cl_release_buffer(svm_atomic_counter);
    }
}
