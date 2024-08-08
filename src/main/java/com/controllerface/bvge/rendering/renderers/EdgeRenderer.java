package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.KernelType;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareEdges_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.PrepareEdges;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;

import static com.controllerface.bvge.game.Constants.Rendering.SCALAR_LENGTH;
import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_2D_LENGTH;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL15C.GL_LINES;
import static org.lwjgl.opengl.GL15C.glDrawArrays;

public class EdgeRenderer extends GameSystem
{
    private static final int DATA_POINTS_PER_EDGE = 2;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_EDGE * VECTOR_2D_LENGTH;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;
    private static final int BATCH_FLAG_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_EDGE * SCALAR_LENGTH;
    private static final int BATCH_FLAG_SIZE = BATCH_FLAG_COUNT * Float.BYTES;
    private static final int EDGE_ATTRIBUTE = 0;
    private static final int FLAG_ATTRIBUTE = 1;

    private final GPUProgram p_prepare_edges = new PrepareEdges();
    private GPUKernel k_prepare_edges;
    private GL_Shader shader;

    private GL_VertexArray vao;
    private GL_VertexBuffer vbo_edge;
    private GL_VertexBuffer vbo_flag;
    private long ptr_vbo_edge;
    private long ptr_vbo_flag;

    public EdgeRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        inti_CL();
    }

    private void init_GL()
    {
        shader = GPU.GL.new_shader("object_outline.glsl", GL_ShaderType.TWO_STAGE);
        vao = GPU.GL.new_vao();
        vbo_edge = GPU.GL.new_buffer_vec2(vao, EDGE_ATTRIBUTE, BATCH_BUFFER_SIZE);
        vbo_flag = GPU.GL.new_buffer_float(vao, FLAG_ATTRIBUTE, BATCH_FLAG_SIZE);
        vao.enable_attribute(EDGE_ATTRIBUTE);
        vao.enable_attribute(FLAG_ATTRIBUTE);
    }

    private void inti_CL()
    {
        p_prepare_edges.init();
        ptr_vbo_edge = GPGPU.share_memory(vbo_edge.id());
        ptr_vbo_flag = GPGPU.share_memory(vbo_flag.id());

        long k_ptr_prepare_edges = p_prepare_edges.kernel_ptr(KernelType.prepare_edges);
        k_prepare_edges = new PrepareEdges_k(GPGPU.ptr_render_queue, k_ptr_prepare_edges)
            .ptr_arg(PrepareEdges_k.Args.vertex_vbo, ptr_vbo_edge)
            .ptr_arg(PrepareEdges_k.Args.flag_vbo, ptr_vbo_flag)
            .buf_arg(PrepareEdges_k.Args.points, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT))
            .buf_arg(PrepareEdges_k.Args.point_hull_indices, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT_HULL_INDEX))
            .buf_arg(PrepareEdges_k.Args.hull_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_FLAG))
            .buf_arg(PrepareEdges_k.Args.edges, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_EDGE))
            .buf_arg(PrepareEdges_k.Args.edge_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_EDGE_FLAG));
    }

    @Override
    public void tick(float dt)
    {
        vao.bind();
        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPGPU.core_memory.last_edge(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPGPU.calculate_preferred_global_size(count);
            k_prepare_edges
                .share_mem(ptr_vbo_edge)
                .share_mem(ptr_vbo_flag)
                .set_arg(PrepareEdges_k.Args.offset, offset)
                .set_arg(PrepareEdges_k.Args.max_edge, count)
                .call(arg_long(count_size), GPGPU.preferred_work_size);

            glDrawArrays(GL_LINES, 0, count * 2);
            offset += count;
        }

        shader.detach();
        vao.unbind();
    }

    @Override
    public void shutdown()
    {
        vao.release();
        vbo_edge.release();
        vbo_flag.release();
        shader.release();
        p_prepare_edges.release();
        GPGPU.cl_release_buffer(ptr_vbo_edge);
        GPGPU.cl_release_buffer(ptr_vbo_flag);
    }
}