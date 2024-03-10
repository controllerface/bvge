package com.controllerface.bvge.gl.renderers;

import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.systems.GameSystem;
import com.controllerface.bvge.gl.AbstractShader;
import com.controllerface.bvge.gl.GLUtils;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Assets;
import com.controllerface.bvge.window.Window;

import static com.controllerface.bvge.util.Constants.Rendering.VECTOR_FLOAT_2D_SIZE;
import static org.lwjgl.opengl.GL11C.glDrawArrays;
import static org.lwjgl.opengl.GL15C.GL_DYNAMIC_DRAW;
import static org.lwjgl.opengl.GL30C.glBindVertexArray;
import static org.lwjgl.opengl.GL45C.*;

public class UniformGridRenderer extends GameSystem
{
    private static final int BUFFER_SIZE = 4 * VECTOR_FLOAT_2D_SIZE;
    private static final int POSITION_ATTRIBUTE = 0;

    private final AbstractShader shader;
    private final UniformGrid uniformGrid;

    private int vao;
    private int point_vbo;

    public UniformGridRenderer(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.shader = Assets.load_shader("bounding_outline.glsl");
        this.uniformGrid = uniformGrid;
        init();
    }

    private void init()
    {
        vao = glCreateVertexArrays();
        point_vbo = GLUtils.new_buffer_vec2(vao, POSITION_ATTRIBUTE, BUFFER_SIZE);
        glEnableVertexArrayAttrib(vao, POSITION_ATTRIBUTE);
    }

    private float[] load_grid_data()
    {
        float[] data = new float[8];
        data[0] = uniformGrid.getX_origin();                        // lower left X
        data[1] = uniformGrid.getY_origin();                        // lower left Y
        data[2] = uniformGrid.getX_origin() + uniformGrid.width;    // lower right X
        data[3] = uniformGrid.getY_origin();                        // lower right Y
        data[4] = uniformGrid.getX_origin() + uniformGrid.width;    // upper right X
        data[5] = uniformGrid.getY_origin() + uniformGrid.height;   // upper right Y
        data[6] = uniformGrid.getX_origin();                        // upper left X
        data[7] = uniformGrid.getY_origin() + uniformGrid.height;   // upper left Y
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
        glDrawArrays(GL_LINE_LOOP, 0, 4);
        glBindVertexArray(0);
        shader.detach();
    }

    @Override
    public void shutdown()
    {
        glDeleteVertexArrays(vao);
        glDeleteBuffers(point_vbo);
    }
}
