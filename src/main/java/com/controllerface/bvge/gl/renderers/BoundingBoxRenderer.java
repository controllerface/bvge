package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.PrepareBounds_k;
import com.controllerface.bvge.cl.programs.PrepareBounds;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import org.lwjgl.system.MemoryUtil;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_2D_LENGTH;
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

    private final int[] offsets = new int[Constants.Rendering.MAX_BATCH_SIZE];
    private final int[] counts = new int[Constants.Rendering.MAX_BATCH_SIZE];

    private final GPUProgram prepare_bounds = new PrepareBounds();

    private int vao;
    private int vbo;
    private long vbo_ptr;

    private Shader shader;
    private GPUKernel prepare_bounds_k;

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
        shader = Assets.load_shader("bounding_outline.glsl");
        vao = glCreateVertexArrays();
        vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
    }

    private void init_CL()
    {
        vbo_ptr = GPGPU.share_memory(vbo);

        prepare_bounds.init();

        long ptr = prepare_bounds.kernel_ptr(Kernel.prepare_bounds);
        prepare_bounds_k = new PrepareBounds_k(GPGPU.command_queue_ptr, ptr)
            .ptr_arg(PrepareBounds_k.Args.vbo, vbo_ptr)
            .buf_arg(PrepareBounds_k.Args.bounds, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_AABB));
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = GPGPU.core_memory.next_hull(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            var offsets = MemoryUtil.memAllocInt(count).put(this.offsets, 0, count).flip();
            var counts = MemoryUtil.memAllocInt(count).put(this.counts, 0, count).flip();

            prepare_bounds_k
                .share_mem(vbo_ptr)
                .set_arg(PrepareBounds_k.Args.offset, offset)
                .call(arg_long(count));

            glMultiDrawArrays(GL_LINE_LOOP, offsets, counts);

            MemoryUtil.memFree(offsets);
            MemoryUtil.memFree(counts);

            offset += count;
        }

        glBindVertexArray(0);

        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo);
        shader.destroy();
        prepare_bounds.destroy();
        GPGPU.cl_release_buffer(vbo_ptr);
    }
}
