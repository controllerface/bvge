package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.RenderBufferType;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.PrepareEdges_k;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.PrepareEdges;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.SCALAR_LENGTH;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_2D_LENGTH;
import static org.lwjgl.opengl.GL15C.GL_LINES;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

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
    private Shader shader;

    private int vao;
    private int vbo_edge;
    private int vbo_flag;
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
        shader = Assets.load_shader("object_outline.glsl");
        vao = glCreateVertexArrays();
        vbo_edge = GLUtils.new_buffer_vec2(vao, EDGE_ATTRIBUTE, BATCH_BUFFER_SIZE);
        vbo_flag = GLUtils.new_buffer_float(vao, FLAG_ATTRIBUTE, BATCH_FLAG_SIZE);
        glEnableVertexArrayAttrib(vao, EDGE_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, FLAG_ATTRIBUTE);
    }

    private void inti_CL()
    {
        p_prepare_edges.init();
        ptr_vbo_edge = GPGPU.share_memory(vbo_edge);
        ptr_vbo_flag = GPGPU.share_memory(vbo_flag);

        long k_ptr_prepare_edges = p_prepare_edges.kernel_ptr(Kernel.prepare_edges);
        k_prepare_edges = new PrepareEdges_k(GPGPU.ptr_render_queue, k_ptr_prepare_edges)
            .ptr_arg(PrepareEdges_k.Args.vertex_vbo, ptr_vbo_edge)
            .ptr_arg(PrepareEdges_k.Args.flag_vbo, ptr_vbo_flag)
            .buf_arg(PrepareEdges_k.Args.points, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT))
            .buf_arg(PrepareEdges_k.Args.edges, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_EDGE))
            .buf_arg(PrepareEdges_k.Args.edge_flags, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_EDGE_FLAG));
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);
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
        glBindVertexArray(0);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo_edge);
        glDeleteBuffers(vbo_flag);
        shader.destroy();
        p_prepare_edges.destroy();
        GPGPU.cl_release_buffer(ptr_vbo_edge);
        GPGPU.cl_release_buffer(ptr_vbo_flag);
    }
}