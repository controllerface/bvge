package com.controllerface.bvge.ecs.systems.renderers.batches;

import com.controllerface.bvge.ecs.Line2D;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import org.joml.Vector2f;
import org.joml.Vector3f;

import java.util.Arrays;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Rendering batch specifically for sprites
 */
public class RectRenderBatch implements Comparable<RectRenderBatch>
{
    private final int POS_SIZE = 3;
    private final int COLOR_SIZE = 3;
    private final int POS_OFFSET = 0;
    private final int COLOR_OFFSET = POS_OFFSET + POS_SIZE * Float.BYTES;
    private final int VERTEX_SIZE = 6;
    private final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;
    private Line2D[] lines;
    private int numLines;
    private boolean hasRoom;
    private float[] vertices;
    private int vaoID, vboID;
    private int zIndex;
    private final AbstractShader currentShader;

    public RectRenderBatch(int zIndex, AbstractShader currentShader)
    {
        this.zIndex = zIndex;
        this.lines = new Line2D[Constants.Rendering.MAX_BATCH_SIZE * 4]; // 4 lines per box

        vertices = new float[Constants.Rendering.MAX_BATCH_SIZE * 4 * VERTEX_SIZE];

        this.numLines = 0;
        this.hasRoom = true;
        //this.textures = new ArrayList<>();
        this.currentShader = currentShader;
    }

    public void clear()
    {
        numLines = 0;
        hasRoom = true;
        Arrays.fill(vertices, 0);
    }

    public void start()
    {
        // Generate and bind a Vertex Array Object
        vaoID = glGenVertexArrays();
        glBindVertexArray(vaoID);

        // Allocate space for vertices
        vboID = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        glBufferData(GL_ARRAY_BUFFER, (long) vertices.length * Float.BYTES, GL_DYNAMIC_DRAW);

        // Enable the buffer attribute pointers
        glVertexAttribPointer(0, POS_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, POS_OFFSET);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, COLOR_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, COLOR_OFFSET);
        glEnableVertexAttribArray(1);
    }

    public void addBox(float x, float y, float w, float h, Vector3f color)
    {
        // Get index and add renderObject
        int index = this.numLines;

        var v1 = new Vector2f(x, y);
        var v2 = new Vector2f(x + w, y);
        var v3 = new Vector2f(x + w, y + h);
        var v4 = new Vector2f(x, y + h);

        var l1 = new Line2D(v1, v2, color);
        var l2 = new Line2D(v2, v3, color);
        var l3 = new Line2D(v3, v4, color);
        var l4 = new Line2D(v4, v1, color);

        this.lines[index] = l1;
        loadVertexProperties(index);
        index++;

        this.lines[index] = l2;
        loadVertexProperties(index);
        index++;

        this.lines[index] = l3;
        loadVertexProperties(index);
        index++;

        this.lines[index] = l4;
        loadVertexProperties(index);
        numLines += 4;

        if (numLines >= Constants.Rendering.MAX_BATCH_SIZE)
        {
            this.hasRoom = false;
        }
    }

    public void render()
    {
        glLineWidth(1.0f);
        for (int i = 0; i < numLines; i++)
        {
            loadVertexProperties(i);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vboID);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertices);

        // Use shader
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.get().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.get().camera().getViewMatrix());

        glBindVertexArray(vaoID);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_LINES, 0, numLines * 6 * 2);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        currentShader.detach();
    }

    /**
     * Updates the local buffer p2 reflect the current state of the indexed sprite.
     *
     * @param index the location of the sprite, within the sprite array
     */
    private void loadVertexProperties(int index)
    {
        Line2D line = this.lines[index];

        // Find offset within array (4 vertices per sprite)
        int offset = index * 2 * VERTEX_SIZE;

        // Load position
        vertices[offset] = line.getFrom().x;
        vertices[offset + 1] = line.getFrom().y;
        vertices[offset + 2] = 0.0f;

        // Load color
        vertices[offset + 3] = line.getColor().x;
        vertices[offset + 4] = line.getColor().y;
        vertices[offset + 5] = line.getColor().z;

        vertices[offset + 6] = line.getTo().x;
        vertices[offset + 7] = line.getTo().y;
        vertices[offset + 8] = 0.0f;

        // Load color
        vertices[offset + 9] = line.getColor().x;
        vertices[offset + 10] = line.getColor().y;
        vertices[offset + 11] = line.getColor().z;
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
    public int compareTo(RectRenderBatch o)
    {
        return Integer.compare(this.zIndex, o.zIndex());
    }
}
