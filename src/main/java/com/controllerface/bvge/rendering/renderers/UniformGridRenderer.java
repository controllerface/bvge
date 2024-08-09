package com.controllerface.bvge.rendering.renderers;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.GameSystem;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.PlayerInput;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexArray;
import com.controllerface.bvge.gpu.gl.buffers.GL_VertexBuffer;
import com.controllerface.bvge.gpu.gl.shaders.GL_Shader;
import com.controllerface.bvge.gpu.gl.shaders.GL_ShaderType;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.rendering.Renderer;

import java.util.Objects;

import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static com.controllerface.bvge.game.Constants.Rendering.VECTOR_FLOAT_4D_SIZE;
import static org.lwjgl.opengl.GL45C.GL_LINE_LOOP;
import static org.lwjgl.opengl.GL45C.glMultiDrawArrays;

public class UniformGridRenderer implements Renderer
{
    private static final int VERTICES_PER_BOX = 4;
    private static final int POSITION_ATTRIBUTE = 0;
    private static final int COLOR_ATTRIBUTE = 1;

    private final UniformGrid uniformGrid;

    private int[] first;
    private int[] count;
    private int vertex_buffer_size;
    private int color_buffer_size;

    private GL_VertexArray vao;
    private GL_VertexBuffer point_vbo;
    private GL_VertexBuffer color_vbo;

    private GL_Shader shader;

    private final ECS ecs;

    private record GridPoint(float x, float y, float r, float g, float b, float a) { }
    private record GridBounds(GridPoint p0, GridPoint p1, GridPoint p2, GridPoint p3) { }

    public UniformGridRenderer(ECS ecs, UniformGrid uniformGrid)
    {
        this.ecs = ecs;
        this.uniformGrid = uniformGrid;
        init_GL();
    }

    private void init_GL()
    {
        int sector_0_count = UniformGrid.BLOCK_COUNT * UniformGrid.BLOCK_COUNT;
        int draw_count = uniformGrid.x_subdivisions * uniformGrid.y_subdivisions + 8 + sector_0_count;
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
        shader = GPU.GL.new_shader("uniform_grid.glsl", GL_ShaderType.TWO_STAGE);
        vao = GPU.GL.new_vao();
        point_vbo = GPU.GL.new_buffer_vec2(vao, POSITION_ATTRIBUTE, vertex_buffer_size);
        color_vbo = GPU.GL.new_buffer_vec4(vao, COLOR_ATTRIBUTE, color_buffer_size);
        vao.enable_attribute(POSITION_ATTRIBUTE);
        vao.enable_attribute(COLOR_ATTRIBUTE);
    }

    private record GridData(float[] vertices, float[] colors) { }

    private boolean within_inner_grid(float a0, float a1, float a2, float a3, float x, float y, float w, float h)
    {
        return a0 < x + w
            && a0 + a2 > x
            && a1 < y + h
            && a1 + a3 > y;
    }

    private int[] write_rect(GridBounds rect, float[] vertex_buffer, float[] color_buffer, int vertex_index, int color_index)
    {
        int v_i = vertex_index;
        int c_i = color_index;
        vertex_buffer[v_i++] = rect.p0.x;  // lower left X
        vertex_buffer[v_i++] = rect.p0.y;  // lower left Y
        vertex_buffer[v_i++] = rect.p1.x;  // lower right X
        vertex_buffer[v_i++] = rect.p1.y;  // lower right Y
        vertex_buffer[v_i++] = rect.p2.x;  // upper right X
        vertex_buffer[v_i++] = rect.p2.y;  // upper right Y
        vertex_buffer[v_i++] = rect.p3.x;  // upper left X
        vertex_buffer[v_i++] = rect.p3.y;  // upper left Y

        color_buffer[c_i++] = rect.p0.r; // v0
        color_buffer[c_i++] = rect.p0.g;
        color_buffer[c_i++] = rect.p0.b;
        color_buffer[c_i++] = rect.p0.a;
        color_buffer[c_i++] = rect.p1.r; // v1
        color_buffer[c_i++] = rect.p1.g;
        color_buffer[c_i++] = rect.p1.b;
        color_buffer[c_i++] = rect.p1.a;
        color_buffer[c_i++] = rect.p2.r; // v2
        color_buffer[c_i++] = rect.p2.g;
        color_buffer[c_i++] = rect.p2.b;
        color_buffer[c_i++] = rect.p2.a;
        color_buffer[c_i++] = rect.p3.r; // v3
        color_buffer[c_i++] = rect.p3.g;
        color_buffer[c_i++] = rect.p3.b;
        color_buffer[c_i++] = rect.p3.a;

        return new int[]{v_i - vertex_index, c_i - color_index};
    }

    private GridData load_grid_data()
    {
        float[] vertex_data = new float[vertex_buffer_size];
        float[] color_data = new float[color_buffer_size];

        int vertex_index = 0;
        int color_index = 0;

        float sector_size = UniformGrid.SECTOR_SIZE;

        float xo = uniformGrid.x_origin();
        float yo = uniformGrid.y_origin();

        float i_xo = uniformGrid.inner_x_origin();
        float i_yo = uniformGrid.inner_y_origin();

        float o_xo = uniformGrid.outer_x_origin();
        float o_yo = uniformGrid.outer_y_origin();

        float s_xo = uniformGrid.sector_origin_x();
        float s_yo = uniformGrid.sector_origin_y();

        var base_bounds_p0 = new GridPoint(xo, yo, 0.5f, 0.5f, 0.5f, 0.5f);
        var base_bounds_p1 = new GridPoint(xo + uniformGrid.width, yo, 0.5f, 0.5f, 0.5f, 0.5f);
        var base_bounds_p2 = new GridPoint(xo + uniformGrid.width, yo + uniformGrid.height, 0.5f, 0.5f, 0.5f, 0.5f);
        var base_bounds_p3 = new GridPoint(xo, yo + uniformGrid.height, 0.5f, 0.5f, 0.5f, 0.5f);
        var base_bounds    = new GridBounds(base_bounds_p0, base_bounds_p1, base_bounds_p2, base_bounds_p3);

        var inner_bounds_p0 = new GridPoint(i_xo, i_yo, 0.5f, 0.5f, 0.5f, 0.5f);
        var inner_bounds_p1 = new GridPoint(i_xo + uniformGrid.inner_width, i_yo, 0.5f, 0.5f, 0.5f, 0.5f);
        var inner_bounds_p2 = new GridPoint(i_xo + uniformGrid.inner_width, i_yo + uniformGrid.inner_height, 0.5f, 0.5f, 0.5f, 0.5f);
        var inner_bounds_p3 = new GridPoint(i_xo, i_yo + uniformGrid.inner_height, 0.5f, 0.5f, 0.5f, 0.5f);
        var inner_bounds    = new GridBounds(inner_bounds_p0, inner_bounds_p1, inner_bounds_p2, inner_bounds_p3);

        var outer_bounds_p0 = new GridPoint(o_xo, o_yo, 0.5f, 0.5f, 0.5f, 0.5f);
        var outer_bounds_p1 = new GridPoint(o_xo + uniformGrid.outer_width, o_yo, 0.5f, 0.5f, 0.5f, 0.5f);
        var outer_bounds_p2 = new GridPoint(o_xo + uniformGrid.outer_width, o_yo + uniformGrid.outer_height, 0.5f, 0.5f, 0.5f, 0.5f);
        var outer_bounds_p3 = new GridPoint(o_xo, o_yo + uniformGrid.outer_height, 0.5f, 0.5f, 0.5f, 0.5f);
        var outer_bounds    = new GridBounds(outer_bounds_p0, outer_bounds_p1, outer_bounds_p2, outer_bounds_p3);

        var sector_bounds_p0 = new GridPoint(s_xo, s_yo, 0.9f, 0.5f, 0.5f, 0.5f);
        var sector_bounds_p1 = new GridPoint(s_xo + uniformGrid.sector_width(), s_yo, 0.9f, 0.5f, 0.5f, 0.5f);
        var sector_bounds_p2 = new GridPoint(s_xo + uniformGrid.sector_width(), s_yo + uniformGrid.sector_height(), 0.9f, 0.5f, 0.5f, 0.5f);
        var sector_bounds_p3 = new GridPoint(s_xo, s_yo + uniformGrid.sector_height(), 0.9f, 0.5f, 0.5f, 0.5f);
        var sector_bounds    = new GridBounds(sector_bounds_p0, sector_bounds_p1, sector_bounds_p2, sector_bounds_p3);

        var sector_0_key = UniformGrid.get_sector_for_point(outer_bounds.p0.x, outer_bounds.p0.y);
        var sector_1_key = UniformGrid.get_sector_for_point(outer_bounds.p1.x, outer_bounds.p1.y);
        var sector_2_key = UniformGrid.get_sector_for_point(outer_bounds.p2.x, outer_bounds.p2.y);
        var sector_3_key = UniformGrid.get_sector_for_point(outer_bounds.p3.x, outer_bounds.p3.y);

        float sector_0_origin_x = (float)sector_0_key[0] * sector_size;
        float sector_0_origin_y = (float)sector_0_key[1] * sector_size;

        float sector_1_origin_x = (float)sector_1_key[0] * sector_size;
        float sector_1_origin_y = (float)sector_1_key[1] * sector_size;

        float sector_2_origin_x = (float)sector_2_key[0] * sector_size;
        float sector_2_origin_y = (float)sector_2_key[1] * sector_size;

        float sector_3_origin_x = (float)sector_3_key[0] * sector_size;
        float sector_3_origin_y = (float)sector_3_key[1] * sector_size;

        var sector_0_p0 = new GridPoint(sector_0_origin_x, sector_0_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_0_p1 = new GridPoint(sector_0_origin_x + sector_size, sector_0_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_0_p2 = new GridPoint(sector_0_origin_x + sector_size, sector_0_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_0_p3 = new GridPoint(sector_0_origin_x, sector_0_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_0    = new GridBounds(sector_0_p0, sector_0_p1, sector_0_p2, sector_0_p3);

        var sector_1_p0 = new GridPoint(sector_1_origin_x, sector_1_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_1_p1 = new GridPoint(sector_1_origin_x + sector_size, sector_1_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_1_p2 = new GridPoint(sector_1_origin_x + sector_size, sector_1_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_1_p3 = new GridPoint(sector_1_origin_x, sector_1_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_1    = new GridBounds(sector_1_p0, sector_1_p1, sector_1_p2, sector_1_p3);

        var sector_2_p0 = new GridPoint(sector_2_origin_x, sector_2_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_2_p1 = new GridPoint(sector_2_origin_x + sector_size, sector_2_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_2_p2 = new GridPoint(sector_2_origin_x + sector_size, sector_2_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_2_p3 = new GridPoint(sector_2_origin_x, sector_2_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_2    = new GridBounds(sector_2_p0, sector_2_p1, sector_2_p2, sector_2_p3);

        var sector_3_p0 = new GridPoint(sector_3_origin_x, sector_3_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_3_p1 = new GridPoint(sector_3_origin_x + sector_size, sector_3_origin_y, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_3_p2 = new GridPoint(sector_3_origin_x + sector_size, sector_3_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_3_p3 = new GridPoint(sector_3_origin_x, sector_3_origin_y + sector_size, 0.9f, 0.5f, 0.5f, 0.2f);
        var sector_3    = new GridBounds(sector_3_p0, sector_3_p1, sector_3_p2, sector_3_p3);

        int[] out;

        out = write_rect(base_bounds, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(inner_bounds, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(outer_bounds, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(sector_bounds, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(sector_0, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(sector_1, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(sector_2, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        out = write_rect(sector_3, vertex_data, color_data, vertex_index, color_index);
        vertex_index += out[0];
        color_index  += out[1];

        PlayerInput player_input = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        assert player_input != null : "Component was null";
        Objects.requireNonNull(player_input);

        var sec = UniformGrid.get_sector_for_point(player_input.get_world_target().x, player_input.get_world_target().y);

        float offset_x = sec[0] * UniformGrid.BLOCK_COUNT;
        float offset_y = sec[1] * UniformGrid.BLOCK_COUNT;

        for (int x = 0; x < UniformGrid.BLOCK_COUNT; x++)
        {
            for (int y = 0; y < UniformGrid.BLOCK_COUNT; y++)
            {
                float r = 0.4f;
                float g = 1.1f;
                float b = 0.1f;
                float a = 0.9f;

                float current_x = ((x+ offset_x) * UniformGrid.BLOCK_SIZE);
                float current_y = ((y+ offset_y) * UniformGrid.BLOCK_SIZE);
                vertex_data[vertex_index++] = current_x;
                vertex_data[vertex_index++] = current_y;
                vertex_data[vertex_index++] = current_x + UniformGrid.BLOCK_SIZE;
                vertex_data[vertex_index++] = current_y;
                vertex_data[vertex_index++] = current_x + UniformGrid.BLOCK_SIZE;
                vertex_data[vertex_index++] = current_y + UniformGrid.BLOCK_SIZE;
                vertex_data[vertex_index++] = current_x;
                vertex_data[vertex_index++] = current_y + UniformGrid.BLOCK_SIZE;

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

        float x_offset = uniformGrid.x_origin();
        float y_offset = uniformGrid.y_origin();
        float inner_x_offset = uniformGrid.inner_x_origin();
        float inner_y_offset = uniformGrid.inner_y_origin();

        for (int x = 0; x < uniformGrid.x_subdivisions; x++)
        {
            float current_x = x * uniformGrid.x_spacing + x_offset;

            for (int y = 0; y < uniformGrid.y_subdivisions; y++)
            {
                float current_y = y * uniformGrid.y_spacing + y_offset;

                boolean inside_inner = within_inner_grid(current_x, current_y,
                    uniformGrid.x_spacing,
                    uniformGrid.y_spacing,
                    inner_x_offset,
                    inner_y_offset,
                    uniformGrid.inner_width,
                    uniformGrid.inner_height);

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

        return new GridData(vertex_data, color_data);
    }

    @Override
    public void render()
    {
        GridData grid_data = load_grid_data();
        vao.bind();
        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());
        point_vbo.load_float_data(grid_data.vertices);
        color_vbo.load_float_data(grid_data.colors);
        glMultiDrawArrays(GL_LINE_LOOP, first, count);
        vao.unbind();
        shader.detach();
    }

    @Override
    public void destroy()
    {
        shader.release();
        vao.release();
        point_vbo.release();
        color_vbo.release();
    }
}
