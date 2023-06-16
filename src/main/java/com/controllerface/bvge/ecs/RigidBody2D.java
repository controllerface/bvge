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

        verts.add(new Point2D(new Vector2f( 100,100), new Vector2f( 100,100)));
        verts.add(new Point2D(new Vector2f( 200,100), new Vector2f( 200,100)));
        verts.add(new Point2D(new Vector2f( 200,100), new Vector2f( 200,100)));
        verts.add(new Point2D(new Vector2f( 100,200), new Vector2f( 100,200)));

        var edges = new ArrayList<Edge2D>();
        var force = 500;

        edges.add(new Edge2D(0, 1));
        edges.add(new Edge2D(1, 2));
        edges.add(new Edge2D(2, 3));
        edges.add(new Edge2D(3, 0));

        edges.add(new Edge2D(0, 2));
        edges.add(new Edge2D(1, 3));

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
