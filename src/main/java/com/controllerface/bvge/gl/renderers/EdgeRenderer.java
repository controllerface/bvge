package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.PrepareEdges_k;
import com.controllerface.bvge.cl.programs.PrepareEdges;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
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

/**
 * Renders physics edge constraints. All defined edges are rendered as lines.
 */
public class EdgeRenderer extends GameSystem
{
    private static final int DATA_POINTS_PER_EDGE = 2;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_EDGE * VECTOR_2D_LENGTH;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;
    private static final int BATCH_FLAG_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_EDGE * SCALAR_LENGTH;
    private static final int BATCH_FLAG_SIZE = BATCH_FLAG_COUNT * Float.BYTES;
    private static final int EDGE_ATTRIBUTE = 0;
    private static final int FLAG_ATTRIBUTE = 1;

    private final AbstractShader shader;
    private final GPUProgram prepare_edges = new PrepareEdges();

    private int vao;
    private int edge_vbo;
    private int flag_vbo;
    private long vertex_vbo_ptr;
    private long flag_vbo_ptr;

    private GPUKernel prepare_edges_k;

    public EdgeRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("object_outline.glsl");
        init_GL();
        inti_CL();
    }

    private void init_GL()
    {
        vao = glCreateVertexArrays();
        edge_vbo = GLUtils.new_buffer_vec2(vao, EDGE_ATTRIBUTE, BATCH_BUFFER_SIZE);
        flag_vbo = GLUtils.new_buffer_float(vao, FLAG_ATTRIBUTE, BATCH_FLAG_SIZE);
        glEnableVertexArrayAttrib(vao, EDGE_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, FLAG_ATTRIBUTE);
    }

    private void inti_CL()
    {
        vertex_vbo_ptr = GPGPU.share_memory(edge_vbo);
        flag_vbo_ptr = GPGPU.share_memory(flag_vbo);

        prepare_edges.init();

        long ptr = prepare_edges.kernel_ptr(Kernel.prepare_edges);
        prepare_edges_k = new PrepareEdges_k(GPGPU.command_queue_ptr, ptr)
            .ptr_arg(PrepareEdges_k.Args.vertex_vbo, vertex_vbo_ptr)
            .ptr_arg(PrepareEdges_k.Args.flag_vbo, flag_vbo_ptr)
            .ptr_arg(PrepareEdges_k.Args.points, GPGPU.Buffer.points.pointer)
            .ptr_arg(PrepareEdges_k.Args.edges, GPGPU.Buffer.edges.pointer)
            .ptr_arg(PrepareEdges_k.Args.edge_flags, GPGPU.Buffer.edge_flags.pointer);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPGPU.core_memory.next_edge(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);

            prepare_edges_k
                .share_mem(vertex_vbo_ptr)
                .share_mem(flag_vbo_ptr)
                .set_arg(PrepareEdges_k.Args.offset, offset)
                .call(arg_long(count));

            glDrawArrays(GL_LINES, 0, count * 2);
            offset += count;
        }

        glBindVertexArray(0);

        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(edge_vbo);
        glDeleteBuffers(flag_vbo);
        prepare_edges.destroy();
        GPGPU.cl_release_buffer(vertex_vbo_ptr);
        GPGPU.cl_release_buffer(flag_vbo_ptr);
    }
}