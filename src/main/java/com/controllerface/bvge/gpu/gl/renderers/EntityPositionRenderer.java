package com.controllerface.bvge.gpu.gl.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareEntities_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.PrepareEntities;
import com.controllerface.bvge.gpu.gl.GLUtils;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.gpu.cl.CLUtils.arg_long;
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
    private GL_Shader shader;

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
        shader = GPU.GL.new_shader("entity_position_shader.glsl", GL_ShaderType.TWO_STAGE);
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
            .buf_arg(PrepareEntities_k.Args.points, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_ENTITY));
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
            int count_size = GPGPU.calculate_preferred_global_size(count);
            k_prepare_entities
                .share_mem(ptr_vbo_vertex)
                .set_arg(PrepareEntities_k.Args.offset, offset)
                .set_arg(PrepareEntities_k.Args.max_entity, count)
                .call(arg_long(count_size), GPGPU.preferred_work_size);

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
        shader.release();
        p_prepare_entities.release();
        GPGPU.cl_release_buffer(ptr_vbo_vertex);
    }
}