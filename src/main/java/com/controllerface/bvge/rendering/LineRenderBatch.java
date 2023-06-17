package com.controllerface.bvge.rendering;

import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import org.joml.Math;
import org.joml.*;
import org.lwjgl.opengl.GL20;

import java.util.ArrayList;
import java.util.Arrays;
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
public class LineRenderBatch
{
    private static int MAX_LINES = 1000;
    private static List<Line2D> lines = new ArrayList<>();

    // 6 floats per vertex, 3 pos, 3 color
    private static float[] vertexArray = new float[MAX_LINES * 6 * 2];

    private static Shader shader = AssetPool.getShader("debugLine2D.glsl");

    private static int vaoId;
    private static int vboId;
    private static boolean started = false;


    public void clear()
    {
        lines.clear();
        //numLines = 0;
        //sprites = new SpriteComponentEX[maxBatchSize * 4 * VERTEX_SIZE];
    }

    public void start()
    {
        // gen VAO
        vaoId = glGenVertexArrays();
        glBindVertexArray(vaoId);

        // create VBO
        vboId = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboId);
        glBufferData(GL_ARRAY_BUFFER, (long) vertexArray.length * Float.BYTES, GL_DYNAMIC_DRAW);

        // enable vertex array attribs

        // these 3 floats are the position
        GL20.glVertexAttribPointer(0, 3, GL_FLOAT, false, 6 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        // thee 3 floats are the color
        GL20.glVertexAttribPointer(1, 3, GL_FLOAT, false, 6 * Float.BYTES, 3 * Float.BYTES);
        glEnableVertexAttribArray(1);

        glLineWidth(2.0f);
    }

    public void addSprite(Line2D line)
    {
        lines.add(line);
        // Get index and add renderObject
//        int index = this.numLines;
//        this.lines[index] = line;
//        this.numLines++;
//
//
//        // Add properties to local vertices array
//        loadVertexProperties(index);
//
//        if (numLines >= Constants.Rendering.MAX_BATCH_SIZE)
//        {
//            this.hasRoom = false;
//        }
    }

    public void render()
    {
        if (lines.size() <= 0)
        {
            return;
        }
        int index = 0;
        for (Line2D line : lines)
        {
            for (int i = 0; i < 2; i++)
            {
                Vector2f pos = i == 0
                    ? line.getFrom()
                    : line.getTo();
                Vector3f color = line.getColor();

                // load pos into array
                vertexArray[index]     = pos.x;
                vertexArray[index + 1] = pos.y;
                vertexArray[index + 2] = -10.0f;

                // load color into array
                vertexArray[index + 3] = color.x;
                vertexArray[index + 4] = color.y;
                vertexArray[index + 5] = color.z;

                index += 6;
            }
        }

        glBindBuffer(GL_ARRAY_BUFFER, vboId);
        glBufferSubData(GL_ARRAY_BUFFER, 0, Arrays.copyOfRange(vertexArray, 0, lines.size() * 6 * 2));

        // use the shader
        shader.use();
        shader.uploadMat4f("uProjection", Window.getScene().camera().getProjectionMatrix());
        shader.uploadMat4f("uView", Window.getScene().camera().getViewMatrix());

        glBindVertexArray(vaoId);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);

        glDrawArrays(GL_LINES, 0, lines.size() * 6 * 2);

        //disable loc
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glBindVertexArray(0);

        // unbind
        shader.detach();
    }

    public boolean hasRoom()
    {
        return lines.size() >= MAX_LINES;
    }


//    @Override
//    public int compareTo(LineRenderBatch o)
//    {
//        return Integer.compare(this.zIndex, o.zIndex());
//    }
}
