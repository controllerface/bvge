package com.controllerface.bvge.ecs;

import org.joml.Vector2f;

import java.util.ArrayList;
import java.util.List;

public class RigidBody2D implements Component_EX
{
    private List<Point2D> verts;
    private List<Edge2D> edges;
    private Vector2f acc =  new Vector2f();
    private float force;

    public RigidBody2D(List<Point2D> verts, List<Edge2D> edges, float force)
    {
        this.verts = verts;
        this.edges = edges;
        this.force = force;
    }

    public static RigidBody2D simpleBox()
    {
        var verts = new ArrayList<Point2D>();
        var edges = new ArrayList<Edge2D>();
        var force = 1000;
        verts.add(new Point2D());
        edges.add(new Edge2D(0, 1));
        return new RigidBody2D(verts, edges, force);
    }

    public List<Point2D> getVerts()
    {
        return verts;
    }

    public List<Edge2D> getEdges()
    {
        return edges;
    }

    public Vector2f getAcc()
    {
        return acc;
    }

    public float getForce()
    {
        return force;
    }

    public void setForce(float force)
    {
        this.force = force;
    }
}
