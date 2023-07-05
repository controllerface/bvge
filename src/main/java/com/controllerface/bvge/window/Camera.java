package com.controllerface.bvge.window;

import org.joml.Matrix4f;
import org.joml.Vector2f;
import org.joml.Vector3f;

public class Camera
{
    float projWidth = 100;
    float projHeight = 100;

    private final Matrix4f projectionMatrix;
    private final Matrix4f viewMatrix;
    private final Matrix4f inverseProjection;
    private final Matrix4f inverseView;
    public Vector2f position;
    public Vector2f projectionSize = new Vector2f(projWidth, projHeight);

    private float zoom = 3f;

    public Camera(Vector2f position)
    {
        this.position = position;
        this.projectionMatrix = new Matrix4f();
        this.viewMatrix = new Matrix4f();
        this.inverseProjection = new Matrix4f();
        this.inverseView = new Matrix4f();
        adjustProjection();
    }

    public void adjustProjection()
    {
        projectionMatrix.identity();
        projectionMatrix.ortho(0.0f,
            projectionSize.x * zoom,
            0.0f,
            projectionSize.y * zoom,
            0.0f,
            100.0f);
        projectionMatrix.invert(inverseProjection);
    }

    public Matrix4f getViewMatrix()
    {
        var cameraFront = new Vector3f(0.0f, 0.0f, -1.0f);
        var cameraUp = new Vector3f(0.0f, 1.0f, 0.0f);
        this.viewMatrix.identity();
        var eye = new Vector3f(position.x, position.y, 20.0f);
        var center = cameraFront.add(position.x, position.y, 0.0f);
        this.viewMatrix.lookAt(eye, center, cameraUp);

        this.viewMatrix.invert(inverseView);

        return this.viewMatrix;
    }

    public Matrix4f getProjectionMatrix()
    {
        return this.projectionMatrix;
    }

    public Matrix4f getInverseProjection()
    {
        return inverseProjection;
    }

    public Matrix4f getInverseView()
    {
        return inverseView;
    }

    public Vector2f getProjectionSize()
    {
        return projectionSize;
    }

    public float getZoom()
    {
        return zoom;
    }

    public void setZoom(float zoom)
    {
        this.zoom = zoom;
    }

    public void addZoom(float value)
    {
        this.zoom += value;
        if (zoom < .5f)
        {
            zoom = .5f;
        }
        else if (zoom > 20f)
        {
            zoom = 20f;
        }
    }
}
