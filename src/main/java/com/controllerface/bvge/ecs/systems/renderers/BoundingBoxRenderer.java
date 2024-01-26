package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import static org.lwjgl.opengl.GL11C.GL_LINE_LOOP;
import static org.lwjgl.opengl.GL11C.GL_FLOAT;
import static org.lwjgl.opengl.GL14C.glMultiDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL15C.glBindBuffer;
import static org.lwjgl.opengl.GL15C.glBufferData;
import static org.lwjgl.opengl.GL15C.glGenBuffers;
import static org.lwjgl.opengl.GL20C.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL30C.glGenVertexArrays;

public class BoundingBoxRenderer extends GameSystem{
    private static final int VERTEX_SIZE = 2;
    private static final int VERTS_PER_EDGE = 2;
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * VERTS_PER_EDGE * VERTEX_SIZE;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;
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
        vao_id = glGenVertexArrays();
        glBindVertexArray(vao_id);

        bounding_box_vbo = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, bounding_box_vbo);
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);
        GPU.share_memory(bounding_box_vbo);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    @Override
    public void tick(float dt)
    {
        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexAttribArray(0);

        int offset = 0;
        for (int remaining = GPU.Memory.next_hull(); remaining > 0; remaining -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, remaining);
            GPU.GL_bounds(bounding_box_vbo, offset, count);
            glMultiDrawArrays(GL_LINE_LOOP, offsets, counts);
            offset += count;
        }

        glDisableVertexAttribArray(0);
        glBindVertexArray(0);

        shader.detach();
    }
}
