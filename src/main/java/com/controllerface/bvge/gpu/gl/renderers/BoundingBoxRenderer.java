package com.controllerface.bvge.gpu.gl.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.cl.GPGPU;
import com.controllerface.bvge.gpu.cl.kernels.GPUKernel;
import com.controllerface.bvge.gpu.cl.kernels.Kernel;
import com.controllerface.bvge.gpu.cl.kernels.rendering.PrepareBounds_k;
import com.controllerface.bvge.gpu.cl.programs.GPUProgram;
import com.controllerface.bvge.gpu.cl.programs.PrepareBounds;
import com.controllerface.bvge.gpu.gl.GLUtils;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.memory.types.RenderBufferType;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_2D_LENGTH;
import static com.controllerface.bvge.gpu.cl.CLUtils.arg_long;
import static org.lwjgl.opengl.GL11C.GL_LINE_LOOP;
import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL15C.glMultiDrawArrays;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class BoundingBoxRenderer extends GameSystem
{
    private static final int DATA_POINTS_PER_BOX = 4;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_BOX * VECTOR_2D_LENGTH;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;
    private static final int POSITION_ATTRIBUTE = 0;

    private final GPUProgram p_prepare_bounds = new PrepareBounds();
    private GPUKernel k_prepare_bounds;
    private GL_Shader shader;

    private int vao;
    private int vbo_position;
    private long ptr_vbo_position;

    private final int[] offsets = new int[Constants.Rendering.MAX_BATCH_SIZE];
    private final int[] counts = new int[Constants.Rendering.MAX_BATCH_SIZE];

    public BoundingBoxRenderer(ECS ecs)
    {
        super(ecs);
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
        vao = glCreateVertexArrays();
        vbo_position = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
    }

    private void init_CL()
    {
        p_prepare_bounds.init();
        ptr_vbo_position = GPGPU.share_memory(vbo_position);

        long k_ptr_prepare_bounds = p_prepare_bounds.kernel_ptr(Kernel.prepare_bounds);
        k_prepare_bounds = new PrepareBounds_k(GPGPU.ptr_render_queue, k_ptr_prepare_bounds)
            .ptr_arg(PrepareBounds_k.Args.vbo, ptr_vbo_position)
            .buf_arg(PrepareBounds_k.Args.bounds, GPGPU.core_memory.get_buffer(RenderBufferType.RENDER_HULL_AABB));
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPGPU.core_memory.last_hull(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            int count_size = GPGPU.calculate_preferred_global_size(count);

            // todo: see if a uniform can be set that sets a max_count, and vertices can bail if beyond max
            int[] o_ = new int[count];
            int[] c_ = new int[count];
            System.arraycopy(this.offsets, 0, o_, 0, count);
            System.arraycopy(this.counts, 0, c_, 0, count);

            k_prepare_bounds
                .share_mem(ptr_vbo_position)
                .set_arg(PrepareBounds_k.Args.offset, offset)
                .set_arg(PrepareBounds_k.Args.max_bound, count)
                .call(arg_long(count_size), GPGPU.preferred_work_size);

            glMultiDrawArrays(GL_LINE_LOOP, o_, c_);

            offset += count;
        }

        shader.detach();
        glBindVertexArray(0);
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo_position);
        shader.release();
        p_prepare_bounds.release();
        GPGPU.cl_release_buffer(ptr_vbo_position);
    }
}
