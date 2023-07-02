package com.controllerface.bvge.gl.batches;

import com.controllerface.bvge.data.FBounds2D;
import com.controllerface.bvge.ecs.Line2D;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import org.joml.Vector2f;
import org.joml.Vector3f;

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
    // Vertex
    // ======
    // Pos               Color                        Tex coords      tex ID
    // float, float,     float, float, float, float   float, float    float
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

    //private int[] texSlots = { 0, 1, 2, 3, 4, 5, 6, 7 };

    //private List<Texture> textures;

    private int vaoID, vboID;

    private int zIndex;

    private final Shader currentShader;

    public RectRenderBatch(int zIndex, Shader currentShader)
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
        //sprites = new SpriteComponentEX[maxBatchSize * 4 * VERTEX_SIZE];
        //textures.clear();
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

        // Create and upload indices buffer
        int eboID = glGenBuffers();
        int[] indices = generateIndices();
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboID);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_DYNAMIC_DRAW);

        // Enable the buffer attribute pointers
        glVertexAttribPointer(0, POS_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, POS_OFFSET);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, COLOR_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, COLOR_OFFSET);
        glEnableVertexAttribArray(1);

//        glVertexAttribPointer(2, TEX_COORDS_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, TEX_COORDS_OFFSET);
//        glEnableVertexAttribArray(2);
//
//        glVertexAttribPointer(3, TEX_ID_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, TEX_ID_OFFSET);
//        glEnableVertexAttribArray(3);
//
//        glVertexAttribPointer(4, ENTITY_ID_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, ENTITY_ID_OFFSET);
//        glEnableVertexAttribArray(4);
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
        index++;
        numLines += 4;


//        if (sprite.getTexture() != null)
//        {
//            if (!textures.contains(sprite.getTexture()))
//            {
//                textures.add(sprite.getTexture());
//            }
//        }

        // Add properties p2 local vertices array
        //loadVertexProperties(index);

        if (numLines >= Constants.Rendering.MAX_BATCH_SIZE)
        {
            this.hasRoom = false;
        }
    }

    public void render()
    {
        //boolean rebuffer= false;
        for (int i = 0; i < numLines; i++)
        {
            //QuadRectangle rend = lines[i];
            //if (rend.isDirty())
            //{
                loadVertexProperties(i);
                //rebuffer = true;
            //}

        }

        // if the sprite data changes in any way, re-buffering is needed.
        // This is actually probably pretty common depending on the types
        // of game objects that are being drawn (animations, etc.)
        //if (rebuffer)
        //{
            glBindBuffer(GL_ARRAY_BUFFER, vboID);
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices);
        //}

        // Use shader
        currentShader.use();
        currentShader.uploadMat4f("uProjection", Window.getScene().camera().getProjectionMatrix());
        currentShader.uploadMat4f("uView", Window.getScene().camera().getViewMatrix());

//        // todo: this is bad, there's no check for the hardware texture slot max
//        //  batches should be grouped by texture, if multiple objects use the same texture,
//        //  they should be preferably batched together
//        for (int i = 0; i < textures.size(); i++)
//        {
//            // todo: actually just set the correct index, bound by the max
//            //  and do -1 in the shader
//            // this + 1 is p2 support using texture 0 as "empty", allowing color p2 take
//            glActiveTexture(GL_TEXTURE0 + i);
//            textures.get(i).bind();
//        }

//        currentShader.uploadIntArray("uTextures", texSlots);

        //glLineWidth(5.0f);

        glBindVertexArray(vaoID);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        glDrawArrays(GL_LINES, 0, numLines * 6 * 2);
        //glDrawElements(GL_TRIANGLES, this.numLines * 6, GL_UNSIGNED_INT, 0);
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

//        for (int i = 0; i < textures.size(); i++)
//        {
//            textures.get(i).unbind();
//        }

        glLineWidth(1.0f);

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

    private int[] generateIndices()
    {
        // 6 indices per quad (3 per triangle)
        int[] elements = new int[6 * Constants.Rendering.MAX_BATCH_SIZE];
        for (int i = 0; i < Constants.Rendering.MAX_BATCH_SIZE; i++)
        {
            loadElementIndices(elements, i);
        }

        return elements;
    }

    private void loadElementIndices(int[] elements, int index)
    {
        int offsetArrayIndex = 6 * index;
        int offset = 4 * index;

        // 3, 2, 0, 0, 2, 1        7, 6, 4, 4, 6, 5
        // Triangle 1
        elements[offsetArrayIndex] = offset + 3;
        elements[offsetArrayIndex + 1] = offset + 2;
        elements[offsetArrayIndex + 2] = offset;

        // Triangle 2
        elements[offsetArrayIndex + 3] = offset;
        elements[offsetArrayIndex + 4] = offset + 2;
        elements[offsetArrayIndex + 5] = offset + 1;
    }

    public boolean hasRoom()
    {
        return this.hasRoom;
    }

//    public boolean hasTextureRoom()
//    {
//        return this.textures.size() < 8;
//    }

//    public boolean hasTexture(Texture texture)
//    {
//        return this.textures.contains(texture);
//    }

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
