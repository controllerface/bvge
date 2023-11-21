package com.controllerface.bvge.ecs.systems.renderers;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opengl.GL11.GL_FLOAT;
import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Manages rendering of edge constraints. All edges that are defined in the currently
 * loaded physics state are rendered as lines.
 */
public class BoneRenderer extends GameSystem
{
    private static final int VERTEX_SIZE = 2; // a vertex is 2 floats (x,y)
    //private static final int VERTS_PER_EDGE = 2;
    private static final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;
    private static final int BATCH_VERTEX_COUNT = Constants.Rendering.MAX_BATCH_SIZE * /* VERTS_PER_EDGE * */ VERTEX_SIZE;
    private static final int BATCH_BUFFER_SIZE = BATCH_VERTEX_COUNT * Float.BYTES;

    private final AbstractShader shader;
    private final List<BoneRenderBatch> batches;
    private int vaoID, vboID;

    public BoneRenderer(ECS ecs)
    {
        super(ecs);
        this.batches = new ArrayList<>();
        this.shader = Assets.shader("bone_shader.glsl");
        start();
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID); // this sets this VAO as being active

        // generate and bind a Vertex Buffer Object, note that because our vao is currently active,
        // the new vbo is "attached" to it implicitly
        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        // allocates enough space for the vertices we can handle in this batch, but doesn't transfer any data
        // into the buffer just yet
        glBufferData(GL_ARRAY_BUFFER, BATCH_BUFFER_SIZE, GL_DYNAMIC_DRAW);

        // share the buffer with the CL context
        GPU.share_memory(vboID);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, VERTEX_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, 0);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    private void render()
    {
        for (BoneRenderBatch batch : batches)
        {
            batch.render();
            batch.clear();
        }
    }

    @Override
    public void tick(float dt)
    {
        // todo: right now, this check only adds batches, never reducing them if the count goes
        //  low enough that some batches would be unneeded. This will leak memory resources
        //  so should be adjusted when deleting entities is added.

        var bone_count = Main.Memory.next_bone();
        var needed_batches = bone_count / Constants.Rendering.MAX_BATCH_SIZE;
        var r = bone_count % Constants.Rendering.MAX_BATCH_SIZE;
        if (r > 0)
        {
            needed_batches++;
        }
        while (needed_batches > batches.size())
        {
            var b = new BoneRenderBatch(shader, vaoID, vboID);
            batches.add(b);
        }

        int next = 0;
        int offset = 0;
        for (int i = bone_count; i > 0; i -= Constants.Rendering.MAX_BATCH_SIZE)
        {
            int count = Math.min(Constants.Rendering.MAX_BATCH_SIZE, i);
            var b = batches.get(next++);
            b.setLineCount(count);
            b.setOffset(offset);
            offset += count;
        }
        render();
    }

    /**
     * Rendering batch specifically for edge constraints
     */
    private static class BoneRenderBatch
    {
        private int numBones;
        private int offset;
        private final int vaoID, vboID;
        private final AbstractShader currentShader;

        public BoneRenderBatch(AbstractShader currentShader, int vaoID, int vboID)
        {
            this.numBones = 0;
            this.currentShader = currentShader;
            this.vaoID = vaoID;
            this.vboID = vboID;
        }

        public void clear()
        {
            numBones = 0;
        }

        public void setLineCount(int numLines)
        {
            this.numBones = numLines;
        }

        public void setOffset(int offset)
        {
            this.offset = offset;
        }

        public void render()
        {
            glBindVertexArray(vaoID);
            glBindBuffer(GL_ARRAY_BUFFER, vboID);

            GPU.GL_bones(vboID, offset, numBones);

            // Use shader
            currentShader.use();
            currentShader.uploadMat4f("uVP", Window.get().camera().get_uVP());

            glEnableVertexAttribArray(0);
            glDrawArrays(GL_POINTS, 0, numBones);
            glDisableVertexAttribArray(0);
            glBindVertexArray(0);

            currentShader.detach();
        }
    }
}