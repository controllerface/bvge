package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes;
import com.controllerface.bvge.gpu.cl.contexts.CL_CommandQueue;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
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
import com.controllerface.bvge.models.geometry.MeshRegistry;
import com.controllerface.bvge.rendering.HullIndexData;
import com.controllerface.bvge.rendering.Renderer;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;

public class CircleRenderer implements Renderer
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
    private CL_Buffer ptr_vbo_transform;
    private CL_Buffer atomic_counter;

    private HullIndexData circle_hulls;

    public CircleRenderer()
    {
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
        ptr_vbo_transform = GPU.CL.gl_share_memory(GPU.compute.context, vbo_transform);
        atomic_counter    = GPU.CL.new_pinned_int(GPU.compute.context);

        k_root_hull_filter = new HullFilter_k(GPU.compute.render_queue, p_root_hull_filter).init();
        k_root_hull_count  = new HullCount_k(GPU.compute.render_queue, p_root_hull_filter).init();

        k_prepare_transforms = new PrepareTransforms_k(GPU.compute.render_queue, p_prepare_transforms)
            .init(ptr_vbo_transform);
    }

    public HullIndexData hull_filter(CL_CommandQueue cmd_queue, int mesh_id)
    {
        GPU.CL.zero_buffer(cmd_queue, atomic_counter, CL_DataTypes.cl_int.size());

        int hull_count = GPU.memory.sector_container().next_hull();
        int hull_size  = GPU.compute.calculate_preferred_global_size(hull_count);

        k_root_hull_count
            .buf_arg(HullCount_k.Args.counter, atomic_counter)
            .set_arg(HullCount_k.Args.mesh_id, mesh_id)
            .set_arg(HullCount_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        int final_count = GPU.CL.read_pinned_int(cmd_queue, atomic_counter);

        if (final_count == 0)
        {
            return new HullIndexData(null, final_count);
        }

        long final_buffer_size = (long) CL_DataTypes.cl_int.size() * final_count;
        var hulls_out = GPU.CL.new_buffer(GPU.compute.context, final_buffer_size);

        GPU.CL.zero_buffer(cmd_queue, atomic_counter, CL_DataTypes.cl_int.size());

        k_root_hull_filter
            .buf_arg(HullFilter_k.Args.hulls_out, hulls_out)
            .buf_arg(HullFilter_k.Args.counter, atomic_counter)
            .set_arg(HullFilter_k.Args.mesh_id, mesh_id)
            .set_arg(HullFilter_k.Args.max_hull, hull_count)
            .call(arg_long(hull_size), GPU.compute.preferred_work_size);

        return new HullIndexData(hulls_out, final_count);
    }

    @Override
    public void render()
    {
        if (circle_hulls != null && circle_hulls.indices() != null)
        {
            circle_hulls.indices().release();
        }

        circle_hulls = hull_filter(GPU.compute.render_queue, MeshRegistry.CIRCLE_MESH);

        if (circle_hulls.count() == 0) return;

        vao.bind();

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPU.compute.calculate_preferred_global_size(count);
            k_prepare_transforms
                .share_mem(ptr_vbo_transform)
                .buf_arg(PrepareTransforms_k.Args.indices, circle_hulls.indices())
                .set_arg(PrepareTransforms_k.Args.offset, offset)
                .set_arg(PrepareTransforms_k.Args.max_hull, count)
                .call(arg_long(count_size), GPU.compute.preferred_work_size);

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        shader.detach();
        vao.unbind();
    }

    @Override
    public void destroy()
    {
        vao.release();
        vbo_transform.release();
        shader.release();
        p_prepare_transforms.release();
        p_root_hull_filter.release();
        ptr_vbo_transform.release();
        atomic_counter.release();
    }
}
