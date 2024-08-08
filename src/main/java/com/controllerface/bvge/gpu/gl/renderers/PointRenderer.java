package com.controllerface.bvge.gpu.gl.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PreparePoints_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.PreparePoints;
import com.controllerface.bvge.gpu.gl.GLUtils;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.cl.CLUtils.arg_long;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class PointRenderer extends GameSystem
{
    private static final int POSITION_BATCH_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COLOR_BATCH_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int COLOR_ATTRIBUTE = 1;

    private final GPUProgram prepare_points = new PreparePoints();

    private GL_VertexArray vao;
    private int vertex_vbo;
    private int color_vbo;
    private long vertex_vbo_ptr;
    private long color_vbo_ptr;

    private GL_Shader shader;
    private GPUKernel k_prepare_points;

    public PointRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = GPU.GL.new_shader("point_shader.glsl", GL_ShaderType.TWO_STAGE);
        vao = GPU.GL.new_vao();
        vertex_vbo = GLUtils.new_buffer_vec2(vao.gl_id(), POSITION_ATTRIBUTE, POSITION_BATCH_SIZE);
        color_vbo = GLUtils.new_buffer_vec4(vao.gl_id(), COLOR_ATTRIBUTE, COLOR_BATCH_SIZE);
        vao.enable_attribute(POSITION_ATTRIBUTE);
        vao.enable_attribute(COLOR_ATTRIBUTE);
    }

    private void init_CL()
    {
        vertex_vbo_ptr = GPGPU.share_memory(vertex_vbo);
        color_vbo_ptr = GPGPU.share_memory(color_vbo);

        prepare_points.init();

        long ptr = prepare_points.kernel_ptr(Kernel.prepare_points);
        k_prepare_points = new PreparePoints_k(GPGPU.ptr_render_queue, ptr)
            .ptr_arg(PreparePoints_k.Args.vertex_vbo, vertex_vbo_ptr)
            .ptr_arg(PreparePoints_k.Args.color_vbo, color_vbo_ptr)
            .buf_arg(PreparePoints_k.Args.anti_gravity, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT_ANTI_GRAV))
            .buf_arg(PreparePoints_k.Args.points, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_POINT));
    }

    @Override
    public void tick(float dt)
    {
        vao.bind();

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glPointSize(5);

        int offset = 0;
        for (int remaining = GPGPU.core_memory.last_point(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPGPU.calculate_preferred_global_size(count);
            k_prepare_points
                .share_mem(vertex_vbo_ptr)
                .share_mem(color_vbo_ptr)
                .set_arg(PreparePoints_k.Args.offset, offset)
                .set_arg(PreparePoints_k.Args.max_point, count)
                .call(arg_long(count_size), GPGPU.preferred_work_size);

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        vao.unbind();
        shader.detach();
    }

    @Override
    public void shutdown()
    {
        vao.release();
        glDeleteBuffers(vertex_vbo);
        shader.release();
        prepare_points.release();
        GPGPU.cl_release_buffer(vertex_vbo_ptr);
    }
}