package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.Shader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class UniformGridRenderer extends GameSystem
{
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int VERTICES_PER_BOX = 4;

    private final UniformGrid uniformGrid;

    private int[] first;
    private int[] count;
    private int buffer_size;

    private int vao;
    private int point_vbo;

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
        buffer_size = draw_count * VERTICES_PER_BOX * VECTOR_FLOAT_2D_SIZE;
        int next = 0;
        for (int i = 0; i < draw_count; i++)
        {
            first[i] = next;
            count[i] = VERTICES_PER_BOX;
            next += VERTICES_PER_BOX;
        }
        shader = Assets.load_shader("uniform_grid.glsl");
        vao = glCreateVertexArrays();
        point_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, buffer_size);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
    }

    private float[] load_grid_data()
    {
        float[] data = new float[buffer_size];
        data[0] = uniformGrid.getX_origin();                        // lower left X
        data[1] = uniformGrid.getY_origin();                        // lower left Y
        data[2] = uniformGrid.getX_origin() + uniformGrid.width;    // lower right X
        data[3] = uniformGrid.getY_origin();                        // lower right Y
        data[4] = uniformGrid.getX_origin() + uniformGrid.width;    // upper right X
        data[5] = uniformGrid.getY_origin() + uniformGrid.height;   // upper right Y
        data[6] = uniformGrid.getX_origin();                        // upper left X
        data[7] = uniformGrid.getY_origin() + uniformGrid.height;   // upper left Y

        int index = 8;
        float x_offset = uniformGrid.getX_origin();
        float y_offset = uniformGrid.getY_origin();
        for (int x = 0; x < uniformGrid.x_subdivisions; x++)
        {
            float current_x = x * uniformGrid.x_spacing + x_offset;
            for (int y = 0; y < uniformGrid.y_subdivisions; y++)
            {
                float current_y = y * uniformGrid.y_spacing + y_offset;
                data[index++] = current_x;
                data[index++] = current_y;
                data[index++] = current_x + uniformGrid.x_spacing;
                data[index++] = current_y;
                data[index++] = current_x + uniformGrid.x_spacing;
                data[index++] = current_y + uniformGrid.y_spacing;
                data[index++] = current_x;
                data[index++] = current_y + uniformGrid.y_spacing;
            }
        }


        return data;
    }

    @Override
    public void tick(float dt)
    {
        float[] data = load_grid_data();

        glBindVertexArray(vao);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glNamedBufferData(point_vbo, data, GL_DYNAMIC_DRAW);
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
