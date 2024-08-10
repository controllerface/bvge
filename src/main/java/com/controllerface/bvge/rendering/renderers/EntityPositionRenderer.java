package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareEntities_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PrepareEntities;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.rendering.Renderer;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL45C.glPointSize;

public class EntityPositionRenderer implements Renderer
{
    private static final int BATCH_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int POSITION_ATTRIBUTE = 0;

    private final GPUProgram p_prepare_entities = new PrepareEntities();
    private GPUKernel k_prepare_entities;
    private GL_Shader shader;

    private GL_VertexArray vao;
    private GL_VertexBuffer vbo_vertex;
    private CL_Buffer ptr_vbo_vertex;

    public EntityPositionRenderer()
    {
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = GPU.GL.new_shader("entity_position_shader.glsl", GL_ShaderType.TWO_STAGE);
        vao = GPU.GL.new_vao();
        vbo_vertex = GPU.GL.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        vao.enable_attribute(POSITION_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_entities.init();
        ptr_vbo_vertex = GPU.CL.gl_share_memory(GPU.compute.context, vbo_vertex);
        k_prepare_entities = new PrepareEntities_k(GPU.compute.render_queue, p_prepare_entities).init(ptr_vbo_vertex);
    }

    @Override
    public void render()
    {
        vao.bind();
        glPointSize(3);
        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPU.memory.last_entity(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPU.compute.calculate_preferred_global_size(count);
            k_prepare_entities
                .share_mem(ptr_vbo_vertex)
                .set_arg(PrepareEntities_k.Args.offset, offset)
                .set_arg(PrepareEntities_k.Args.max_entity, count)
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
        vbo_vertex.release();
        shader.release();
        p_prepare_entities.release();
        ptr_vbo_vertex.release();
    }
}