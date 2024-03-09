package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import org.lwjgl.system.MemoryStack;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_2D_LENGTH;
import static org.lwjgl.opengl.GL11C.GL_LINE_LOOP;
import static org.lwjgl.opengl.GL15C.*;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;
import static org.lwjgl.opengl.GL45C.*;

public class BoundingBoxRenderer extends GameSystem
{
    private static final int DATA_POINTS_PER_BOX = 4;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * DATA_POINTS_PER_BOX * VECTOR_2D_LENGTH;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;

    private static final int POSITION_ATTRIBUTE = 0;

    private final AbstractShader shader;
    private int vao_id;
    private int bounding_box_vbo;
    private final int[] offsets = new int[Constants.Rendering.MAX_BATCH_SIZE];
    private final int[] counts = new int[Constants.Rendering.MAX_BATCH_SIZE];

    public BoundingBoxRenderer(ECS ecs)
    {
        super(ecs);
        this.shader = Assets.load_shader("bounding_outline.glsl");
        for (int i = 0; i < Constants.Rendering.MAX_BATCH_SIZE; i++)
        {
            offsets[i] = i * 4;
            counts[i] = 4;
        }
        init();
    }

    public void init()
    {
        vao_id = glCreateVertexArrays();
        bounding_box_vbo = GLUtils.new_buffer_vec2(vao_id, POSITION_ATTRIBUTE, BATCH_BUFFER_SIZE);
        GPU.share_memory(bounding_box_vbo);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);

        int offset = 0;
        for (int remaining = GPU.core_memory.next_hull(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            try (var mem_stack = MemoryStack.stackPush())
            {
                var offsets = mem_stack.mallocInt(count).put(this.offsets, 0, count).flip();
                var counts = mem_stack.mallocInt(count).put(this.counts, 0, count).flip();
                GPU.GL_bounds(bounding_box_vbo, offset, count);
                glMultiDrawArrays(GL_LINE_LOOP, offsets, counts);
            }
            offset += count;
        }

        glDisableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);

        glBindVertexArray(0);

        shader.detach();
    }
}
