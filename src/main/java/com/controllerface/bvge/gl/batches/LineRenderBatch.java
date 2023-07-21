package com.controllerface.bvge.gl.batches;

import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.Random;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Rendering batch specifically for sprites
 */
public class LineRenderBatch implements Comparable<LineRenderBatch>
{
    private final int POS_SIZE = 3;
    private final int COLOR_SIZE = 3;

    private final int POS_OFFSET = 0;
    private final int COLOR_OFFSET = POS_OFFSET + POS_SIZE * Float.BYTES;

    private final int VERTEX_SIZE = 6;
    private final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;

    private FEdge2D[] lines;
    private int numLines;
    private boolean hasRoom;
    private float[] vertices;

    // todo: try and make two vbo's and avoid needed to deal with offsets when defining attribute
    //  pointers.
    private int vaoID, vboID;

    private int zIndex;

    private final Shader currentShader;

    public LineRenderBatch(int zIndex, Shader currentShader)
    {
        this.zIndex = zIndex;
        this.lines = new FEdge2D[Constants.Rendering.MAX_BATCH_SIZE];

        vertices = new float[Constants.Rendering.MAX_BATCH_SIZE * 2 * VERTEX_SIZE];

        this.numLines = 0;
        this.hasRoom = true;
        this.currentShader = currentShader;
    }

    public void clear()
    {
        numLines = 0;
        this.hasRoom = true;
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
        glBufferData(GL_ARRAY_BUFFER, (long) vertices.length * Float.BYTES, GL_DYNAMIC_DRAW);

        // define the buffer attribute pointers
        glVertexAttribPointer(0, POS_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, POS_OFFSET);
        glVertexAttribPointer(1, COLOR_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, COLOR_OFFSET);

        // unbind the vao since we're done defining things for now
        glBindVertexArray(0);
    }

    public void addLine(FEdge2D line)
    {
        // Get index and add renderObject
        int index = this.numLines;
        this.lines[index] = line;
        this.numLines++;

        // Add properties p2 local vertices array
        loadVertexProperties(index);

        if (numLines >= Constants.Rendering.MAX_BATCH_SIZE)
        {
            this.hasRoom = false;
        }
    }

    public void render()
    {
        glLineWidth(2.0f);
        glBindVertexArray(vaoID);
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices);

        // Use shader
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_LINES, 0, numLines * 6 * 2);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        currentShader.detach();
    }


    private Random rand = new Random();

    /**
     * Updates the local buffer p2 reflect the current state of the indexed sprite.
     *
     * @param index the location of the sprite, within the sprite array
     */
    private void loadVertexProperties(int index)
    {
        float r = 0f; //rand.nextFloat() / 7.0f;
        float g = 0f; //rand.nextFloat() / 5.0f;
        float b = 0f; //rand.nextFloat() / 3.0f;

        FEdge2D line = this.lines[index];

        // Find offset within array (4 vertices per sprite)
        int offset = index * 2 * VERTEX_SIZE;

        // Load position
        vertices[offset] = line.p1().pos_x();
        vertices[offset + 1] = line.p1().pos_y();
        vertices[offset + 2] = 0.0f;
        vertices[offset + 3] = r;
        vertices[offset + 4] = g;
        vertices[offset + 5] = b;

        vertices[offset + 6] = line.p2().pos_x();
        vertices[offset + 7] = line.p2().pos_y();
        vertices[offset + 8] = 0.0f;
        vertices[offset + 9] = r;
        vertices[offset + 10] = g;
        vertices[offset + 11] = b;
    }

    public boolean hasRoom()
    {
        return this.hasRoom;
    }

    public int zIndex()
    {
        return this.zIndex;
    }

    @Override
    public int compareTo(LineRenderBatch o)
    {
        return Integer.compare(this.zIndex, o.zIndex());
    }
}
