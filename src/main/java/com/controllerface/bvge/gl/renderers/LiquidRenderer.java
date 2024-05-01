package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.*;
import com.controllerface.bvge.cl.kernels.PrepareLiquids_k;
import com.controllerface.bvge.cl.kernels.PrepareTransforms_k;
import com.controllerface.bvge.cl.programs.PrepareLiquids;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.ModelRegistry;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.MAX_BATCH_SIZE;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glEnableVertexArrayAttrib;

public class LiquidRenderer extends GameSystem
{
    public static final int CIRCLES_BUFFER_SIZE = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;
    private static final int COLOR_BUFFER_SIZE  = MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;

    private static final int TRANSFORM_ATTRIBUTE = 0;
    private static final int COLOR_ATTRIBUTE     = 1;

    private final GPUProgram prepare_liquids = new PrepareLiquids();

    private int vao;
    private int vbo;
    private int vcb;
    private long vbo_ptr;
    private long color_buffer_ptr;

    private HullIndexData circle_hulls;

    private Shader shader;
    private GPUKernel prepare_liquids_k;

    public LiquidRenderer(ECS ecs)
    {
        super(ecs);
        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        shader = Assets.load_shader("water_shader.glsl");
        vao = glCreateVertexArrays();
        vbo = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, CIRCLES_BUFFER_SIZE);

        vcb = GLUtils.new_buffer_vec4(vao, COLOR_ATTRIBUTE, COLOR_BUFFER_SIZE);

        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, COLOR_ATTRIBUTE);
    }

    private void init_CL()
    {
        vbo_ptr = GPGPU.share_memory(vbo);
        color_buffer_ptr = GPGPU.share_memory(vcb);
        prepare_liquids.init();

        long ptr = prepare_liquids.kernel_ptr(Kernel.prepare_liquids);
        prepare_liquids_k = (new PrepareLiquids_k(GPGPU.command_queue_ptr, ptr))
            .ptr_arg(PrepareLiquids_k.Args.transforms_out, vbo_ptr)
            .ptr_arg(PrepareLiquids_k.Args.colors_out, color_buffer_ptr)
            .buf_arg(PrepareLiquids_k.Args.hull_positions, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL))
            .buf_arg(PrepareLiquids_k.Args.hull_scales, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_SCALE))
            .buf_arg(PrepareLiquids_k.Args.hull_rotations, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_ROTATION))
            .buf_arg(PrepareLiquids_k.Args.hull_point_tables, GPGPU.core_memory.buffer(BufferType.MIRROR_HULL_POINT_TABLE))
            .buf_arg(PrepareLiquids_k.Args.point_hit_counts, GPGPU.core_memory.buffer(BufferType.MIRROR_POINT_HIT_COUNT));
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != -1)
        {
            GPGPU.cl_release_buffer(circle_hulls.indices());
        }
        circle_hulls = GPGPU.GL_hull_filter(ModelRegistry.CIRCLE_PARTICLE);

        if (circle_hulls.count() == 0) return;

        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);

            prepare_liquids_k
                .share_mem(vbo_ptr)
                .share_mem(color_buffer_ptr)
                .ptr_arg(PrepareLiquids_k.Args.indices, circle_hulls.indices())
                .set_arg(PrepareLiquids_k.Args.offset, offset)
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
        glDeleteBuffers(vbo);
        shader.destroy();
        prepare_liquids.destroy();
        GPGPU.cl_release_buffer(vbo_ptr);
    }
}
