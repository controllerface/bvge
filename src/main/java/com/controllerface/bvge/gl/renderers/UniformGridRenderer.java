package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class UniformGridRenderer extends GameSystem
{
    private static final int VERTICES_PER_BOX = 4;

    private static final int POSITION_ATTRIBUTE = 0;
    private static final int COLOR_ATTRIBUTE = 1;

    private final UniformGrid uniformGrid;

    private int[] first;
    private int[] count;
    private int vertex_buffer_size;
    private int color_buffer_size;

    private int vao;
    private int point_vbo;
    private int color_vbo;

    private Shader shader;

    public UniformGridRenderer(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        init_GL();
    }

    private void init_GL()
    {
        int draw_count = uniformGrid.x_subdivisions * uniformGrid.y_subdivisions + 1;
        first = new int[draw_count];
        count = new int[draw_count];
        vertex_buffer_size = draw_count * VERTICES_PER_BOX * VECTOR_FLOAT_2D_SIZE;
        color_buffer_size = draw_count * VERTICES_PER_BOX * VECTOR_FLOAT_4D_SIZE;
        int next = 0;
        for (int i = 0; i < draw_count; i++)
        {
            first[i] = next;
            count[i] = VERTICES_PER_BOX;
            next += VERTICES_PER_BOX;
        }
        shader = Assets.load_shader("uniform_grid.glsl");
        vao = glCreateVertexArrays();
        point_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, vertex_buffer_size);
        color_vbo = GLUtils.new_buffer_vec4(vao, COLOR_ATTRIBUTE, color_buffer_size);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
        glEnableVertexArrayAttrib(vao, COLOR_ATTRIBUTE);
    }

    private record GridData(float[] vertices, float[] colors) {}

    private boolean once = false;

    private boolean inside_inner(float a0, float a1, float a2, float a3, float x, float y, float w, float h)
    {
        return a0 < x + w
            && a0 + a2 > x
            && a1 < y + h
            && a1 + a3 > y;
    }

    private GridData load_grid_data()
    {
        float[] vertex_data = new float[vertex_buffer_size];
        float[] color_data = new float[color_buffer_size];
        vertex_data[0] = uniformGrid.getX_origin();                        // lower left X
        vertex_data[1] = uniformGrid.getY_origin();                        // lower left Y
        vertex_data[2] = uniformGrid.getX_origin() + uniformGrid.width;    // lower right X
        vertex_data[3] = uniformGrid.getY_origin();                        // lower right Y
        vertex_data[4] = uniformGrid.getX_origin() + uniformGrid.width;    // upper right X
        vertex_data[5] = uniformGrid.getY_origin() + uniformGrid.height;   // upper right Y
        vertex_data[6] = uniformGrid.getX_origin();                        // upper left X
        vertex_data[7] = uniformGrid.getY_origin() + uniformGrid.height;   // upper left Y

        color_data[0]  = 0.5f; // v0
        color_data[1]  = 0.5f;
        color_data[2]  = 0.5f;
        color_data[3]  = 0.5f;
        color_data[4]  = 0.5f; // v1
        color_data[5]  = 0.5f;
        color_data[6]  = 0.5f;
        color_data[7]  = 0.5f;
        color_data[8]  = 0.5f; // v2
        color_data[9]  = 0.5f;
        color_data[10] = 0.5f;
        color_data[11] = 0.5f;
        color_data[12] = 0.5f; // v3
        color_data[13] = 0.5f;
        color_data[14] = 0.5f;
        color_data[15] = 0.5f;

        int vertex_index = 8;
        int color_index = 16;

        float x_offset = uniformGrid.getX_origin();
        float y_offset = uniformGrid.getY_origin();
        float inner_x_offset = x_offset + (uniformGrid.width - uniformGrid.inner_width) / 2;
        float inner_y_offset = y_offset + (uniformGrid.height - uniformGrid.inner_height) / 2;

        for (int x = 0; x < uniformGrid.x_subdivisions; x++)
        {
            float current_x = x * uniformGrid.x_spacing + x_offset;

            for (int y = 0; y < uniformGrid.y_subdivisions; y++)
            {
                float current_y = y * uniformGrid.y_spacing + y_offset;

                boolean inside_inner = inside_inner(current_x, current_y,
                    uniformGrid.x_spacing,
                    uniformGrid.y_spacing,
                    inner_x_offset,
                    inner_y_offset,
                    uniformGrid.inner_width,
                    uniformGrid.inner_height);

                if (!once)
                {
                    System.out.println(STR."ring ding!\{current_x} :: \{current_y}");
                }

                float r = inside_inner
                    ? 0.0f
                    : 0.4f;

                float g = inside_inner
                    ? 0.0f
                    : 0.1f;

                float b = inside_inner
                    ? 0.4f
                    : 0.1f;

                float a = inside_inner
                    ? 0.1f
                    : 0.2f;

                vertex_data[vertex_index++] = current_x;
                vertex_data[vertex_index++] = current_y;
                vertex_data[vertex_index++] = current_x + uniformGrid.x_spacing;
                vertex_data[vertex_index++] = current_y;
                vertex_data[vertex_index++] = current_x + uniformGrid.x_spacing;
                vertex_data[vertex_index++] = current_y + uniformGrid.y_spacing;
                vertex_data[vertex_index++] = current_x;
                vertex_data[vertex_index++] = current_y + uniformGrid.y_spacing;

                color_data[color_index++] = r; // v0
                color_data[color_index++] = g;
                color_data[color_index++] = b;
                color_data[color_index++] = a;
                color_data[color_index++] = r; // v1
                color_data[color_index++] = g;
                color_data[color_index++] = b;
                color_data[color_index++] = a;
                color_data[color_index++] = r; // v2
                color_data[color_index++] = g;
                color_data[color_index++] = b;
                color_data[color_index++] = a;
                color_data[color_index++] = r; // v3
                color_data[color_index++] = g;
                color_data[color_index++] = b;
                color_data[color_index++] = a;
            }
        }


        once = true;

        return new GridData(vertex_data, color_data);
    }

    @Override
    public void tick(float dt)
    {
        GridData grid_data = load_grid_data();

        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glNamedBufferData(point_vbo, grid_data.vertices, GL_DYNAMIC_DRAW);
        glNamedBufferData(color_vbo, grid_data.colors, GL_DYNAMIC_DRAW);
        glMultiDrawArrays(GL_LINE_LOOP, first, count);
        glBindVertexArray(0);
        shader.detach();
    }

    @Override
    public void shutdown()
    {
        shader.destroy();
        glDeleteVertexArrays(vao);
        glDeleteBuffers(point_vbo);
    }
}
