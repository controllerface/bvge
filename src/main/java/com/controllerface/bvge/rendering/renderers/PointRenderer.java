package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PreparePoints_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PreparePoints;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.rendering.Renderer;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDrawArrays;
import static org.lwjgl.opengl.GL45C.glPointSize;

public class PointRenderer implements Renderer
{
    private static final int POSITION_BATCH_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_2D_SIZE;
    private static final int COLOR_BATCH_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int COLOR_ATTRIBUTE = 1;

    private final GPUProgram prepare_points = new PreparePoints();

    private GL_VertexArray vao;
    private GL_VertexBuffer vertex_vbo;
    private GL_VertexBuffer color_vbo;
    private CL_Buffer vertex_buf;
    private CL_Buffer color_buf;

    private GL_Shader shader;
    private GPUKernel k_prepare_points;

    public PointRenderer()
    {
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = GPU.GL.new_shader("point_shader.glsl", GL_ShaderType.TWO_STAGE);
        vao = GPU.GL.new_vao();
        vertex_vbo = GPU.GL.new_buffer_vec2(vao, POSITION_ATTRIBUTE, POSITION_BATCH_SIZE);
        color_vbo = GPU.GL.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BATCH_SIZE);
        vao.enable_attribute(POSITION_ATTRIBUTE);
        vao.enable_attribute(COLOR_ATTRIBUTE);
    }

    private void init_CL()
    {
        vertex_buf = GPU.CL.gl_share_memory(GPU.compute.context, vertex_vbo);
        color_buf =GPU.CL.gl_share_memory(GPU.compute.context, color_vbo);
        prepare_points.init();
        k_prepare_points = new PreparePoints_k(GPU.compute.render_queue, prepare_points).init(vertex_buf, color_buf);
    }

    @Override
    public void render()
    {
        vao.bind();

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glPointSize(5);

        int offset = 0;
        for (int remaining = GPU.memory.last_point(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPU.compute.calculate_preferred_global_size(count);
            k_prepare_points
                .share_mem(vertex_buf)
                .share_mem(color_buf)
                .set_arg(PreparePoints_k.Args.offset, offset)
                .set_arg(PreparePoints_k.Args.max_point, count)
                .call(arg_long(count_size), GPU.compute.preferred_work_size);

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        vao.unbind();
        shader.detach();
    }

    @Override
    public void destroy()
    {
        vao.release();
        vertex_vbo.release();
        shader.release();
        prepare_points.release();
        vertex_buf.release();
        color_buf.release();
    }
}