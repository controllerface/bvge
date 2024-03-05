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

    private final UniformGrid uniformGrid;
    private final float x_offset;
    private final float y_offset;

    private int vao_id;
    private int point_vbo;

    private final AbstractShader shader;

    public UniformGridRenderer(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.shader = Assets.load_shader("bounding_outline.glsl");
        this.uniformGrid = uniformGrid;
        this.x_offset = uniformGrid.width / 2;
        this.y_offset = uniformGrid.height / 2;
        init();
    }

    private void init()
    {
        vao_id = glCreateVertexArrays();
        point_vbo = GLUtils.new_buffer_vec2(vao_id, POSITION_ATTRIBUTE, BUFFER_SIZE);
    }

    @Override
    public void tick(float dt)
    {
        float[] data = new float[8];
        data[0] = uniformGrid.getX_origin();
        data[1] = uniformGrid.getY_origin();

        data[2] = uniformGrid.getX_origin() + uniformGrid.width;
        data[3] = uniformGrid.getY_origin();

        data[4] = uniformGrid.getX_origin() + uniformGrid.width;
        data[5] = uniformGrid.getY_origin() + uniformGrid.height;

        data[6] = uniformGrid.getX_origin();
        data[7] = uniformGrid.getY_origin() + uniformGrid.height;

        glBindVertexArray(vao_id);

        shader.use();
        shader.uploadMat4f("uVP", Window.get().camera().get_uVP());

        glEnableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glNamedBufferData(point_vbo, data, GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_LOOP, 0, 4);
        glDisableVertexArrayAttrib(vao_id, POSITION_ATTRIBUTE);
        glBindVertexArray(0);
        shader.detach();
    }
}
