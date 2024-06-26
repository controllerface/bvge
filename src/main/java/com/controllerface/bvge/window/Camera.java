package com.controllerface.bvge.window;

import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector3f;

public class Camera
{
    private static final float MAX_ZOOM = 2f;
    private static final float MIN_ZOOM = .5f;

    private final Vector2f position;
    private final Vector2f projection_size = new Vector2f(1, 1);
    private final Matrix4f projection_matrix;
    private final Matrix4f view_matrix;
    private final Matrix4f uVP;

    private float zoom = 2f;

    int width;
    int height;

    public Camera(Vector2f position, int height, int width)
    {
        this.height   = height;
        this.width    = width;
        this.position = position;

        this.projection_matrix = new Matrix4f();
        this.view_matrix = new Matrix4f();
        this.uVP              = new Matrix4f();

        adjust_projection(this.height, this.width);
    }

    public Vector2f position()
    {
        return position;
    }

    public Vector2f projection_size()
    {
        return projection_size;
    }

    public void adjust_position(float x, float y)
    {
        position.x = x;
        position.y = y;
    }

    public void adjust_projection(int height, int width)
    {
        this.height = height;
        this.width =  width;

        projection_matrix.identity();

        projection_matrix.ortho(0.0f,
            projection_size.x * zoom,
            0.0f,
            projection_size.y * zoom,
            -6.0f,
            6.0f);

        projection_matrix.mul(get_view_matrix(), uVP);
    }

    private Matrix4f get_view_matrix()
    {
        var cameraFront = new Vector3f(0.0f, 0.0f, -1.0f);
        var cameraUp = new Vector3f(0.0f, 1.0f, 0.0f);
        this.view_matrix.identity();
        var eye = new Vector3f(position.x, position.y, 0.0f);
        var center = cameraFront.add(position.x, position.y, 0.0f);
        this.view_matrix.lookAt(eye, center, cameraUp);
        return this.view_matrix;
    }

    /**
     * This is the view-projection matrix used in uniform values within shaders. It is required
     * to translate world space objects to view space for rendering.
     *
     * @return the current view-projection matrix
     */
    public Matrix4f get_uVP()
    {
        return uVP;
    }

    public float get_zoom()
    {
        return zoom;
    }

    public void add_zoom(float value)
    {
        this.zoom += value;
        if (zoom < MIN_ZOOM)
        {
            zoom = MIN_ZOOM;
        }
        else if (zoom > MAX_ZOOM)
        {
            zoom = MAX_ZOOM;
        }
    }
}
