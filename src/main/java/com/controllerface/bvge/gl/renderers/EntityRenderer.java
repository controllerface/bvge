package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.PrepareEntities_k;
import com.controllerface.bvge.cl.programs.PrepareEntities;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

/**
 * Renders physics edge constraints. All defined edges are rendered as lines.
 */
public class EntityRenderer extends GameSystem
{
    private static final int BATCH_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int POSITION_ATTRIBUTE = 0;

    private final GPUProgram prepare_entities = new PrepareEntities();

    private int vao;
    private int vertex_vbo;
    private long vertex_vbo_ptr;

    private Shader shader;
    private GPUKernel prepare_entities_k;

    public EntityRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = Assets.load_shader("entity_shader.glsl");
        vao = glCreateVertexArrays();
        vertex_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
    }

    private void init_CL()
    {
        vertex_vbo_ptr = GPGPU.share_memory(vertex_vbo);

        prepare_entities.init();

        long ptr = prepare_entities.kernel_ptr(Kernel.prepare_entities);
        prepare_entities_k = new PrepareEntities_k(GPGPU.gl_cmd_queue_ptr, ptr)
            .ptr_arg(PrepareEntities_k.Args.vertex_vbo, vertex_vbo_ptr)
            .buf_arg(PrepareEntities_k.Args.points, GPGPU.core_memory.buffer(BufferType.MIRROR_ENTITY));
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glPointSize(3);

        int offset = 0;
        for (int remaining = GPGPU.core_memory.last_entity(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);

            prepare_entities_k
                .share_mem(vertex_vbo_ptr)
                .set_arg(PrepareEntities_k.Args.offset, offset)
                .call(arg_long(count));

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        glBindVertexArray(0);

        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vertex_vbo);
        shader.destroy();
        prepare_entities.destroy();
        GPGPU.cl_release_buffer(vertex_vbo_ptr);
    }
}