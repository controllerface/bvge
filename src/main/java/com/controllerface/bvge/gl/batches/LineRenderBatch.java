package com.controllerface.bvge.gl.batches;

import com.controllerface.bvge.data.FEdge2D;
import com.controllerface.bvge.ecs.Edge2D;
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

    private FEdge2D[] lines;
    private int numLines;
    private boolean hasRoom;
    private float[] vertices;

    //private int[] texSlots = { 0, 1, 2, 3, 4, 5, 6, 7 };

    //private List<Texture> textures;

    private int vaoID, vboID;

    private int zIndex;

    private final Shader currentShader;

    public LineRenderBatch(int zIndex, Shader currentShader)
    {
        this.zIndex = zIndex;
        this.lines = new FEdge2D[Constants.Rendering.MAX_BATCH_SIZE];

        // 4 vertices for quads, sprites are always rectangular
        vertices = new float[Constants.Rendering.MAX_BATCH_SIZE * 2 * VERTEX_SIZE];

        this.numLines = 0;
        this.hasRoom = true;
        //this.textures = new ArrayList<>();
        this.currentShader = currentShader;
    }

    public void clear()
    {
        numLines = 0;
        this.hasRoom = true;
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

    public void addLine(FEdge2D line)
    {
        // Get index and add renderObject
        int index = this.numLines;
        this.lines[index] = line;
        this.numLines++;

//        if (sprite.getTexture() != null)
//        {
//            if (!textures.contains(sprite.getTexture()))
//            {
//                textures.add(sprite.getTexture());
//            }
//        }

        // Add properties p2 local vertices array
        loadVertexProperties(index);

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
            //Line2D rend = lines[i];
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
        float r = rand.nextFloat() / 7.0f;
        float g = rand.nextFloat() / 5.0f;
        float b = rand.nextFloat() / 3.0f;

        FEdge2D line = this.lines[index];

        // Find offset within array (4 vertices per sprite)
        int offset = index * 2 * VERTEX_SIZE;

        // Load position
        vertices[offset] = line.p1().pos_x();
        vertices[offset + 1] = line.p1().pos_y();
        vertices[offset + 2] = 0.0f;

        // Load color
        vertices[offset + 3] = r;
        vertices[offset + 4] = g;
        vertices[offset + 5] = b;


        vertices[offset + 6] = line.p2().pos_x();
        vertices[offset + 7] = line.p2().pos_y();
        vertices[offset + 8] = 0.0f;

        // Load color
        vertices[offset + 9] = r;
        vertices[offset + 10] = g;
        vertices[offset + 11] = b;
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
    public int compareTo(LineRenderBatch o)
    {
        return Integer.compare(this.zIndex, o.zIndex());
    }
}
