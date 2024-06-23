package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.CoreBufferType;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.PrepareEntities_k;
import com.controllerface.bvge.cl.programs.GPUProgram;
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

public class EntityPositionRenderer extends GameSystem
{
    private static final int BATCH_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int POSITION_ATTRIBUTE = 0;

    private final GPUProgram p_prepare_entities = new PrepareEntities();
    private GPUKernel k_prepare_entities;
    private Shader shader;

    private int vao;
    private int vbo_vertex;
    private long ptr_vbo_vertex;

    public EntityPositionRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = Assets.load_shader("entity_position_shader.glsl");
        vao = glCreateVertexArrays();
        vbo_vertex = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_entities.init();
        ptr_vbo_vertex = GPGPU.share_memory(vbo_vertex);

        long k_ptr_prepare_entities = p_prepare_entities.kernel_ptr(Kernel.prepare_entities);
        k_prepare_entities = new PrepareEntities_k(GPGPU.ptr_render_queue, k_ptr_prepare_entities)
            .ptr_arg(PrepareEntities_k.Args.vertex_vbo, ptr_vbo_vertex)
            .buf_arg(PrepareEntities_k.Args.points, GPGPU.core_memory.get_buffer(CoreBufferType.MIRROR_ENTITY));
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);
        glPointSize(3);
        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPGPU.core_memory.last_entity(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);

            k_prepare_entities
                .share_mem(ptr_vbo_vertex)
                .set_arg(PrepareEntities_k.Args.offset, offset)
                .call(arg_long(count));

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        shader.detach();
        glBindVertexArray(0);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo_vertex);
        shader.destroy();
        p_prepare_entities.destroy();
        GPGPU.cl_release_buffer(ptr_vbo_vertex);
    }
}