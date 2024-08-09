package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.buffers.CL_Buffer;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareBounds_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.rendering.PrepareBounds;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;
import com.controllerface.bvge.rendering.Renderer;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_2D_LENGTH;
import static com.controllerface.bvge.gpu.GPU.CL.arg_long;
import static org.lwjgl.opengl.GL11C.GL_LINE_LOOP;
import static org.lwjgl.opengl.GL15C.glMultiDrawArrays;

public class BoundingBoxRenderer implements Renderer
{
    private static final int DATA_POINTS_PER_BOX = 4;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_BOX * VECTOR_2D_LENGTH;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;
    private static final int POSITION_ATTRIBUTE = 0;

    private final GPUProgram p_prepare_bounds = new PrepareBounds();
    private GPUKernel k_prepare_bounds;
    private GL_Shader shader;

    private GL_VertexArray vao;
    private GL_VertexBuffer vbo_position;
    private CL_Buffer ptr_vbo_position;

    private final int[] offsets = new int[Constants.Rendering.MAX_BATCH_SIZE];
    private final int[] counts = new int[Constants.Rendering.MAX_BATCH_SIZE];

    public BoundingBoxRenderer()
    {
        for (int i = 0; i < Constants.Rendering.MAX_BATCH_SIZE; i++)
        {
            offsets[i] = i * 4;
            counts[i] = 4;
        }
        init_GL();
        init_CL();
    }

    private void init_GL()
    {
        shader = GPU.GL.new_shader("bounding_outline.glsl", GL_ShaderType.TWO_STAGE);
        vao = GPU.GL.new_vao();
        vbo_position = GPU.GL.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        vao.enable_attribute(POSITION_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_bounds.init();
        ptr_vbo_position = GPU.CL.gl_share_memory(GPU.compute.context, vbo_position);

        k_prepare_bounds = new PrepareBounds_k(GPU.compute.render_queue, p_prepare_bounds)
            .buf_arg(PrepareBounds_k.Args.vbo, ptr_vbo_position)
            .buf_arg(PrepareBounds_k.Args.bounds, GPU.memory.get_buffer(RenderBufferType.RENDER_HULL_AABB));
    }

    @Override
    public void render()
    {
        vao.bind();

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPU.memory.last_hull(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPU.compute.calculate_preferred_global_size(count);

            // todo: see if a uniform can be set that sets a max_count, and vertices can bail if beyond max
            int[] o_ = new int[count];
            int[] c_ = new int[count];
            System.arraycopy(this.offsets, 0, o_, 0, count);
            System.arraycopy(this.counts, 0, c_, 0, count);

            k_prepare_bounds
                .share_mem(ptr_vbo_position)
                .set_arg(PrepareBounds_k.Args.offset, offset)
                .set_arg(PrepareBounds_k.Args.max_bound, count)
                .call(arg_long(count_size), GPU.compute.preferred_work_size);

            glMultiDrawArrays(GL_LINE_LOOP, o_, c_);

            offset += count;
        }

        shader.detach();
        vao.unbind();
    }

    @Override
    public void destroy()
    {
        vao.release();
        vbo_position.release();
        shader.release();
        p_prepare_bounds.release();
        ptr_vbo_position.release();
    }
}
