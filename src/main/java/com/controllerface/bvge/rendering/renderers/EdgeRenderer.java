package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareEdges_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PrepareEdges;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.rendering.Renderer;

import static com.controllerface.bvge.game.Constants.Rendering.SCALAR_LENGTH;
import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_2D_LENGTH;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL15C.GL_LINES;
import static org.lwjgl.opengl.GL15C.glDrawArrays;

public class EdgeRenderer implements Renderer
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
    private CL_Buffer ptr_vbo_edge;
    private CL_Buffer ptr_vbo_flag;

    public EdgeRenderer()
    {
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
        ptr_vbo_edge = GPU.CL.gl_share_memory(GPU.compute.context, vbo_edge);
        ptr_vbo_flag = GPU.CL.gl_share_memory(GPU.compute.context, vbo_flag);
        k_prepare_edges = new PrepareEdges_k(GPU.compute.render_queue, p_prepare_edges).init(ptr_vbo_edge, ptr_vbo_flag);
    }

    @Override
    public void render()
    {
        vao.bind();
        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPU.memory.last_edge(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPU.compute.calculate_preferred_global_size(count);
            k_prepare_edges
                .share_mem(ptr_vbo_edge)
                .share_mem(ptr_vbo_flag)
                .set_arg(PrepareEdges_k.Args.offset, offset)
                .set_arg(PrepareEdges_k.Args.max_edge, count)
                .call(arg_long(count_size), GPU.compute.preferred_work_size);

            glDrawArrays(GL_LINES, 0, count * 2);
            offset += count;
        }

        shader.detach();
        vao.unbind();
    }

    @Override
    public void destroy()
    {
        vao.release();
        vbo_edge.release();
        vbo_flag.release();
        shader.release();
        p_prepare_edges.release();
        ptr_vbo_edge.release();
        ptr_vbo_flag.release();
    }
}