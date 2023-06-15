package com.controllerface.bvge.rendering;

import com.controllerface.bvge.window.Window;
import com.controllerface.bvge.util.AssetPool;
import com.controllerface.bvge.util.JMath;
import org.joml.Vector2f;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;

// todo: maybe move this usage into a more generic renderer
//  it currently kind of duplicates the renderer code
public class DebugDraw
{
    private static int MAX_LINES = 500;
    private static List<Line2D> lines = new ArrayList<>();

    // 6 floats per vertex, 3 pos, 3 color
    private static float[] vertexArray = new float[MAX_LINES * 6 * 2];

    private static Shader shader = AssetPool.getShader("debugLine2D.glsl");

    private static int vaoId;
    private static int vboId;
    private static boolean started = false;

    public static void start()
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
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 6 * Float.BYTES, 0);
        glEnableVertexAttribArray(0);

        // thee 3 floats are the color
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 6 * Float.BYTES, 3 * Float.BYTES);
        glEnableVertexAttribArray(1);

        glLineWidth(2.0f);
    }

    public static void beginFrame()
    {
        if (!started)
        {
            start();
            started = true;
        }

        // remove old lines
        for (int i = 0; i < lines.size(); i++)
        {
            if (lines.get(i).beginFrame() < 0)
            {
                lines.remove(i);
                i--;
            }
        }
    }

    public static void draw()
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

    public static void addLine2D(Vector2f from, Vector2f to)
    {
        // todo: add some common constants for arguments
        addLine2D(from, to, new Vector3f(0,1,0), 1);
    }

    public static void addLine2D(Vector2f from, Vector2f to, Vector3f color)
    {
        addLine2D(from, to, color, 1);
    }

    // add line 2d method
    public static void addLine2D(Vector2f from, Vector2f to, Vector3f color, int lifetime)
    {
        if (lines.size() >= MAX_LINES)
        {
            return;
        }
        DebugDraw.lines.add(new Line2D(to, from, color, lifetime));
    }

    // add box methods

    public static void addBox2D(Vector2f center, Vector2f dimensions, float rotation)
    {
        addBox2D(center, dimensions, new Vector3f(0,1,0), rotation);
    }

    public static void addBox2D(Vector2f center, Vector2f dimensions, Vector3f color, float rotation)
    {
        addBox2D(center, dimensions, color, rotation, 1);
    }

    public static void addBox2D(Vector2f center, Vector2f dimensions, Vector3f color,
                                float rotation, int lifetime)
    {
        Vector2f min = new Vector2f(center).sub(new Vector2f(dimensions).mul(0.5f));
        Vector2f max = new Vector2f(center).add(new Vector2f(dimensions).mul(0.5f));

        Vector2f[] verts  =
            {
                new Vector2f(min.x, min.y),
                new Vector2f(min.x, max.y),
                new Vector2f(max.x, max.y),
                new Vector2f(max.x, min.y)
            };

        if (rotation != 0.0)
        {
            for (Vector2f vert : verts)
            {
                JMath.rotate(vert, rotation, center);
            }
        }

        addLine2D(verts[0], verts[1], color, lifetime);
        addLine2D(verts[0], verts[3], color, lifetime);
        addLine2D(verts[1], verts[2], color, lifetime);
        addLine2D(verts[2], verts[3], color, lifetime);

    }

    // add circles

    public static void addCircle2D(Vector2f center, float radius)
    {
        addCircle2D(center, radius, new Vector3f(0,1,0));
    }

    public static void addCircle2D(Vector2f center, float radius, Vector3f color)
    {
        addCircle2D(center, radius, color, 1);
    }


    public static void addCircle2D(Vector2f center, float radius, Vector3f color, int lifetime)
    {
        Vector2f[] points = new Vector2f[20];
        int increment = 360 / points.length;
        int currentAngle = 0;

        for (int i = 0; i < points.length; i++)
        {
            Vector2f tmp = new Vector2f(radius, 0);
            JMath.rotate(tmp, currentAngle, new Vector2f());
            points[i] = new Vector2f(tmp).add(center);

            if (i > 0)
            {
                addLine2D(points[i - 1], points[i], color, lifetime);
            }
            currentAngle += increment;
        }

        addLine2D(points[points.length-1], points[0], color, lifetime);
    }


}
