package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.buffers.MirrorBufferType;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.PreparePoints_k;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.cl.programs.PreparePoints;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
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

    private int vao;
    private int vertex_vbo;
    private int color_vbo;
    private long vertex_vbo_ptr;
    private long color_vbo_ptr;

    private Shader shader;
    private GPUKernel k_prepare_points;

    public PointRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = Assets.load_shader("point_shader.glsl");
        vao = glCreateVertexArrays();
        vertex_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, POSITION_BATCH_SIZE);
        color_vbo = GLUtils.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BATCH_SIZE);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, COLOR_ATTRIBUTE);
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
            .buf_arg(PreparePoints_k.Args.anti_gravity, GPGPU.core_memory.get_buffer(MirrorBufferType.MIRROR_POINT_ANTI_GRAV))
            .buf_arg(PreparePoints_k.Args.points, GPGPU.core_memory.get_buffer(MirrorBufferType.MIRROR_POINT));
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);

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

        glBindVertexArray(0);

        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vertex_vbo);
        shader.destroy();
        prepare_points.destroy();
        GPGPU.cl_release_buffer(vertex_vbo_ptr);
    }
}