package com.controllerface.bvge.ecs.components;

import com.controllerface.bvge.ecs.Edge2D;
import com.controllerface.bvge.ecs.Point2D;
import org.joml.Vector2f;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class RigidBody2D implements GameComponent
{
    private String entitiy;
    private List<Point2D> verts;
    private List<Edge2D> edges;
    private Vector2f acc =  new Vector2f();
    private float force;

    public RigidBody2D(List<Point2D> verts, List<Edge2D> edges, float force, String entitiy)
    {
        this.verts = verts;
        this.edges = edges;
        this.force = force;
        this.entitiy = entitiy;
    }

    public static RigidBody2D simpleBox(float x, float y, float size, String entitiy)
    {
        var verts = new ArrayList<Point2D>();

        var halfSize = size / 2;

        var v1 = new Vector2f(x - halfSize, y - halfSize);
        var v2 = new Vector2f(x + halfSize, y - halfSize);
        var v3 = new Vector2f(x + halfSize, y + halfSize);
        var v4 = new Vector2f(x - halfSize, y + halfSize);

        var p1 = new Point2D(v1, new Vector2f(v1));
        var p2 = new Point2D(v2, new Vector2f(v2));
        var p3 = new Point2D(v3, new Vector2f(v3));
        var p4 = new Point2D(v4, new Vector2f(v4));

        verts.add(p1);
        verts.add(p2);
        verts.add(p3);
        verts.add(p4);

        var edges = new ArrayList<Edge2D>();

        // sides of the box
        edges.add(new Edge2D(p1, p2));
        edges.add(new Edge2D(p2, p3));
        edges.add(new Edge2D(p3, p4));
        edges.add(new Edge2D(p4, p1));

        // corner to corner braces
        edges.add(new Edge2D(p1, p3));
        edges.add(new Edge2D(p2, p4));

        var force = 500;

        return new RigidBody2D(verts, edges, force, entitiy);
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

    public String getEntitiy()
    {
        return entitiy;
    }

    @Override
    public boolean equals(Object o)
    {
        if (this == o)
        {
            return true;
        }
        if (o == null || getClass() != o.getClass())
        {
            return false;
        }
        RigidBody2D that = (RigidBody2D) o;
        return Objects.equals(entitiy, that.entitiy);
    }

    @Override
    public int hashCode()
    {
        int result = entitiy != null
            ? entitiy.hashCode()
            : 0;;
        return result;
    }
}
