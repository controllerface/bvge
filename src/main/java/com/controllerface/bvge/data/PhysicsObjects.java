package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

public class PhysicsObjects
{
    private static Vector2f vectorBuffer = new Vector2f();

    public static FBody2D simpleBox(float x, float y, float size, String entity)
    {
        var halfSize = size / 2;

        var p1 = Main.Memory.newPoint(x - halfSize, y - halfSize);
        var p2 = Main.Memory.newPoint(x + halfSize, y - halfSize);
        var p3 = Main.Memory.newPoint(x + halfSize, y + halfSize);
        var p4 = Main.Memory.newPoint(x - halfSize, y + halfSize);

        var points = new FPoint2D[]{ p1, p2, p3, p4 };

        // box sides
        var e1 = Main.Memory.newEdge(
            p1.index() / Main.Memory.Width.POINT,
            p2.index() / Main.Memory.Width.POINT,
            p2.distance(p1),
            p1, p2);

        var e2 = Main.Memory.newEdge(
            p2.index() / Main.Memory.Width.POINT,
            p3.index() / Main.Memory.Width.POINT,
            p3.distance(p2),
            p2, p3);

        var e3 = Main.Memory.newEdge(
            p3.index() / Main.Memory.Width.POINT,
            p4.index() / Main.Memory.Width.POINT,
            p4.distance(p3),
            p3, p4);

        var e4 = Main.Memory.newEdge(
            p4.index() / Main.Memory.Width.POINT,
            p1.index() / Main.Memory.Width.POINT,
            p1.distance(p4),
            p4, p1);

        // corner braces
        var e5 = Main.Memory.newEdge(
            p1.index() / Main.Memory.Width.POINT,
            p3.index() / Main.Memory.Width.POINT,
            p3.distance(p1),
            p1, p3);

        var e6 = Main.Memory.newEdge(
            p2.index() / Main.Memory.Width.POINT,
            p4.index() / Main.Memory.Width.POINT,
            p4.distance(p2),
            p2, p4);

        var edges = new FEdge2D[]{ e1, e2, e3, e4, e5, e6 };

        var force = 500;

        var bounds = Main.Memory.newBounds();

        MathEX.centroid(points, vectorBuffer);

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size, 0,0,
            p1.index() / Main.Memory.Width.POINT,
            p4.index() / Main.Memory.Width.POINT,
            e1.index() / Main.Memory.Width.EDGE,
            e6.index() / Main.Memory.Width.EDGE,
            bounds.index() / Main.Memory.Width.BOUNDS,
            points, edges, bounds, force, entity);
    }
}
