package com.controllerface.bvge.rendering;

import com.controllerface.bvge.window.Window;
import org.joml.Math;
import org.joml.*;

import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.glDisableVertexAttribArray;
import static org.lwjgl.opengl.GL20.glEnableVertexAttribArray;
import static org.lwjgl.opengl.GL20C.glVertexAttribPointer;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

/**
 * Rendering batch specifically for sprites
 */
public class SpriteRenderBatchEX implements Comparable<SpriteRenderBatchEX>
{
    // Vertex
    // ======
    // Pos               Color                        Tex coords      tex ID
    // float, float,     float, float, float, float   float, float    float
    private final int POS_SIZE = 2;
    private final int COLOR_SIZE = 4;
    private final int TEX_COORDS_SIZE = 2;
    private final int TEX_ID_SIZE = 1;
    private final int ENTITY_ID_SIZE = 1;

    private final int POS_OFFSET = 0;
    private final int COLOR_OFFSET = POS_OFFSET + POS_SIZE * Float.BYTES;
    private final int TEX_COORDS_OFFSET = COLOR_OFFSET + COLOR_SIZE * Float.BYTES;
    private final int TEX_ID_OFFSET = TEX_COORDS_OFFSET + TEX_COORDS_SIZE * Float.BYTES;
    private final int ENTITY_ID_OFFSET = TEX_ID_OFFSET + TEX_ID_SIZE * Float.BYTES;
    private final int VERTEX_SIZE = 10;
    private final int VERTEX_SIZE_BYTES = VERTEX_SIZE * Float.BYTES;

    private SpriteComponentEX[] sprites;
    private int numSprites;
    private boolean hasRoom;
    private float[] vertices;

    private int[] texSlots = { 0, 1, 2, 3, 4, 5, 6, 7 };

    private List<Texture> textures;

    private int vaoID, vboID;
    private int maxBatchSize;

    private int zIndex;

    private final Shader currentShader;

    public SpriteRenderBatchEX(int maxBatchSize, int zIndex, Shader currentShader)
    {
        this.zIndex = zIndex;
        this.sprites = new SpriteComponentEX[maxBatchSize];
        this.maxBatchSize = maxBatchSize;

        // 4 vertices for quads, sprites are always rectangular
        vertices = new float[maxBatchSize * 4 * VERTEX_SIZE];

        this.numSprites = 0;
        this.hasRoom = true;
        this.textures = new ArrayList<>();
        this.currentShader = currentShader;
    }

    public void clear()
    {
        numSprites = 0;
        //sprites = new SpriteComponentEX[maxBatchSize * 4 * VERTEX_SIZE];
        textures.clear();
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
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

        // Enable the buffer attribute pointers
        glVertexAttribPointer(0, POS_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, POS_OFFSET);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, COLOR_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, COLOR_OFFSET);
        glEnableVertexAttribArray(1);

        glVertexAttribPointer(2, TEX_COORDS_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, TEX_COORDS_OFFSET);
        glEnableVertexAttribArray(2);

        glVertexAttribPointer(3, TEX_ID_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, TEX_ID_OFFSET);
        glEnableVertexAttribArray(3);

        glVertexAttribPointer(4, ENTITY_ID_SIZE, GL_FLOAT, false, VERTEX_SIZE_BYTES, ENTITY_ID_OFFSET);
        glEnableVertexAttribArray(4);
    }

    public void addSprite(SpriteComponentEX sprite)
    {
        // Get index and add renderObject
        int index = this.numSprites;
        this.sprites[index] = sprite;
        this.numSprites++;

        if (sprite.getTexture() != null)
        {
            if (!textures.contains(sprite.getTexture()))
            {
                textures.add(sprite.getTexture());
            }
        }

        // Add properties to local vertices array
        loadVertexProperties(index);

        if (numSprites >= this.maxBatchSize)
        {
            this.hasRoom = false;
        }
    }

    public void render()
    {
        boolean rebuffer= false;
        for (int i = 0; i < numSprites; i++)
        {
            SpriteComponentEX rend = sprites[i];
            if (rend.isDirty())
            {
                loadVertexProperties(i);
                rend.setClean();
                rebuffer = true;
            }

        }

        // if the sprite data changes in any way, re-buffering is needed.
        // This is actually probably pretty common depending on the types
        // of game objects that are being drawn (animations, etc.)
        if (rebuffer)
        {
            glBindBuffer(GL_ARRAY_BUFFER, vboID);
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices);
        }

        // Use shader
        Shader shader = currentShader;
        shader.uploadMat4f("uProjection", Window.getScene().camera().getProjectionMatrix());
        shader.uploadMat4f("uView", Window.getScene().camera().getViewMatrix());

        // todo: this is bad, there's no check for the hardware texture slot max
        for (int i = 0; i < textures.size(); i++)
        {
            // todo: actually just set the correct index, bound by the max
            //  and do -1 in the shader
            // this + 1 is to support using texture 0 as "empty", allowing color to take
            glActiveTexture(GL_TEXTURE0 + i + 1);
            textures.get(i).bind();
        }

        shader.uploadIntArray("uTextures", texSlots);

        glBindVertexArray(vaoID);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glDrawElements(GL_TRIANGLES, this.numSprites * 6, GL_UNSIGNED_INT, 0);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);


        for (int i =0; i< textures.size(); i++)
        {
            textures.get(i).unbind();
        }

        shader.detach();
    }

    /**
     * Updates the local buffer to reflect the current state of the indexed sprite.
     *
     * @param index the location of the sprite, within the sprite array
     */
    private void loadVertexProperties(int index)
    {
        SpriteComponentEX sprite = this.sprites[index];

        // Find offset within array (4 vertices per sprite)
        int offset = index * 4 * VERTEX_SIZE;

        Vector4f color = sprite.getColor();
        Vector2f[] texCoords = sprite.getTexCoords();

        int texId = 0;
        if (sprite.getTexture() != null)
        {
            for (int i = 0; i < textures.size(); i++)
            {
                if (textures.get(i).equals(sprite.getTexture()))
                {
                    texId = i + 1; // this + 1 is to support using texture 0 as "empty"
                    break;
                }
            }
        }

        boolean isRotated = sprite.transform.rotation != 0.0f;
        Matrix4f transformMatrix = new Matrix4f().identity();
        if (isRotated)
        {
            transformMatrix.translate(sprite.transform.position.x,
                sprite.transform.position.y,0.0f);

            transformMatrix.rotate(Math.toRadians(sprite.transform.rotation),
                new Vector3f(0,0,1));

            transformMatrix.scale(sprite.transform.scale.x,
                sprite.transform.scale.y, 1.0f);
        }

        // Add vertices with the appropriate properties
        float xAdd = 1.0f;
        float yAdd = 1.0f;
        for (int i = 0; i < 4; i++)
        {
            if (i == 1)
            {
                yAdd = 0.0f;
            }
            else if (i == 2)
            {
                xAdd = 0.0f;
            }
            else if (i == 3)
            {
                yAdd = 1.0f;
            }

            Vector4f currentPos =new Vector4f(
                sprite.transform.position.x + (xAdd * sprite.transform.scale.x),
                sprite.transform.position.y + (yAdd * sprite.transform.scale.y),
                0, 1);

            if (isRotated)
            {
                currentPos = new Vector4f(xAdd, yAdd, 0, 1 ).mul(transformMatrix);
            }

            // Load position
            vertices[offset] = currentPos.x;
            vertices[offset + 1] = currentPos.y;

            // Load color
            vertices[offset + 2] = color.x;
            vertices[offset + 3] = color.y;
            vertices[offset + 4] = color.z;
            vertices[offset + 5] = color.w;

            // Load UV
            vertices[offset + 6] = texCoords[i].x;
            vertices[offset + 7] = texCoords[i].y;

            // text ID
            vertices[offset + 8] = texId;

            // entity ID
            vertices[offset + 9] = -1; //sprite.gameObject.getUid() + 1;
            // todo: add back in a UID system for this, it is used for picking

            offset += VERTEX_SIZE;
        }
    }

    private int[] generateIndices()
    {
        // 6 indices per quad (3 per triangle)
        int[] elements = new int[6 * maxBatchSize];
        for (int i = 0; i < maxBatchSize; i++)
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
        elements[offsetArrayIndex + 2] = offset + 0;

        // Triangle 2
        elements[offsetArrayIndex + 3] = offset + 0;
        elements[offsetArrayIndex + 4] = offset + 2;
        elements[offsetArrayIndex + 5] = offset + 1;
    }

    public boolean hasRoom()
    {
        return this.hasRoom;
    }

    public boolean hasTextureRoom()
    {
        return this.textures.size() < 8;
    }

    public boolean hasTexture(Texture texture)
    {
        return this.textures.contains(texture);
    }

    public int zIndex()
    {
        return this.zIndex;
    }

    @Override
    public int compareTo(SpriteRenderBatchEX o)
    {
        return Integer.compare(this.zIndex, o.zIndex());
    }
}
