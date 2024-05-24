package com.controllerface.bvge.window;

import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector3f;

public class Camera
{
    private final Matrix4f projectionMatrix;
    private final Matrix4f viewMatrix;
    private final Matrix4f uVP;
    public Vector2f position;
    public Vector2f projectionSize = new Vector2f(1, 1);

    private static final float MAX_ZOOM = 2.5f;
    private static final float MIN_ZOOM = 1f;

    private float zoom = 2f;

    int width, height;

    public Camera(Vector2f position, int height, int width)
    {
        this.height = height;
        this.width = width;
        this.position = position;
        this.projectionMatrix = new Matrix4f();
        this.viewMatrix = new Matrix4f();
        this.uVP = new Matrix4f();
        adjustProjection(this.height, this.width);
    }

    public void adjustProjection(int height, int width)
    {
        this.height = height;
        this.width =  width;

        projectionMatrix.identity();

        projectionMatrix.ortho(0.0f,
            projectionSize.x * zoom,
            0.0f,
            projectionSize.y * zoom,
            -6.0f,
            6.0f);

        projectionMatrix.mul(getViewMatrix(), uVP);
    }

    public Matrix4f getViewMatrix()
    {
        var cameraFront = new Vector3f(0.0f, 0.0f, -1.0f);
        var cameraUp = new Vector3f(0.0f, 1.0f, 0.0f);
        this.viewMatrix.identity();
        var eye = new Vector3f(position.x, position.y, 0.0f);
        var center = cameraFront.add(position.x, position.y, 0.0f);
        this.viewMatrix.lookAt(eye, center, cameraUp);
        return this.viewMatrix;
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
