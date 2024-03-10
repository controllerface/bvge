package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.cl.GPUKernel;
import com.controllerface.bvge.cl.GPUProgram;
import com.controllerface.bvge.cl.HullIndexData;
import com.controllerface.bvge.cl.kernels.PrepareTransforms_k;
import com.controllerface.bvge.cl.programs.PrepareTransforms;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.geometry.Models;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.cl.CLUtils.arg_long;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opencl.CL12.clReleaseMemObject;
import static org.lwjgl.opengl.ARBDirectStateAccess.glCreateVertexArrays;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_POINTS;
import static org.lwjgl.opengl.GL15C.glDeleteBuffers;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glDeleteVertexArrays;
import static org.lwjgl.opengl.GL45C.glEnableVertexArrayAttrib;

public class CircleRenderer extends GameSystem
{
    public static final int CIRCLES_BUFFER_SIZE = Constants.Rendering.MAX_BATCH_SIZE * VECTOR_FLOAT_4D_SIZE;

    private static final int TRANSFORM_ATTRIBUTE = 0;

    private final AbstractShader shader;
    private final GPUProgram prepare_transforms = new PrepareTransforms();

    private int vao;
    private int vbo;
    private long vbo_ptr;

    private HullIndexData circle_hulls;

    private GPUKernel prepare_transforms_k;

    public CircleRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("circle_shader.glsl");
        init_GL();
        init_CL();
    }

    public void init_GL()
    {
        vao = glCreateVertexArrays();
        vbo = GLUtils.new_buffer_vec4(vao, TRANSFORM_ATTRIBUTE, CIRCLES_BUFFER_SIZE);
        vbo_ptr = GPU.share_memory(vbo);
        glEnableVertexArrayAttrib(vao, TRANSFORM_ATTRIBUTE);
    }

    private void init_CL()
    {
        prepare_transforms.init();

        long ptr = prepare_transforms.kernel_ptr(GPU.Kernel.prepare_transforms);
        prepare_transforms_k = (new PrepareTransforms_k(GPU.command_queue_ptr, ptr))
            .mem_arg(PrepareTransforms_k.Args.transforms, GPU.Buffer.hulls.memory)
            .mem_arg(PrepareTransforms_k.Args.hull_rotations, GPU.Buffer.hull_rotation.memory);
    }

    @Override
    public void tick(float dt)
    {
        if (circle_hulls != null && circle_hulls.indices() != -1)
        {
            clReleaseMemObject(circle_hulls.indices());
        }
        circle_hulls = GPU.GL_hull_filter(Models.CIRCLE_PARTICLE);

        if (circle_hulls.count() == 0) return;

        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        int offset = 0;
        for (int remaining = circle_hulls.count(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);

            prepare_transforms_k
                .share_mem(vbo_ptr)
                .ptr_arg(PrepareTransforms_k.Args.indices, circle_hulls.indices())
                .ptr_arg(PrepareTransforms_k.Args.transforms_out, vbo_ptr)
                .set_arg(PrepareTransforms_k.Args.offset, offset)
                .call(arg_long(count));

            glDrawArrays(GL_POINTS, 0, count);
            offset += count;
        }

        GPU.cl_release_buffer(circle_hulls.indices());

        glBindVertexArray(0);
        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(vbo);
        prepare_transforms.destroy();
        GPU.cl_release_buffer(vbo_ptr);
    }
}
