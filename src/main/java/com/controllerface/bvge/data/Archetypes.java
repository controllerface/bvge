package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

public class Archetypes
{
    private static Vector2f vectorBuffer = new Vector2f();

    public static FBody2D simpleBox(float x, float y, float size, String entity)
    {
        var halfSize = size / 2;

        var v1 = Main.Memory.newPoint(x + halfSize, y - halfSize);
        var v2 = Main.Memory.newPoint(x + halfSize, y - halfSize);
        var v3 = Main.Memory.newPoint(x + halfSize, y + halfSize);
        var v4 = Main.Memory.newPoint(x - halfSize, y + halfSize);

        var verts = new FPoint2D[]{ v1, v2, v3, v4 };

        // box sides
        var e1 = Main.Memory.newEdge(
            v1.index() / Main.Memory.Width.POINT,
            v2.index() / Main.Memory.Width.POINT,
            v2.distance(v1),
            v1, v2);

        var e2 = Main.Memory.newEdge(
            v2.index() / Main.Memory.Width.POINT,
            v3.index() / Main.Memory.Width.POINT,
            v3.distance(v2),
            v2, v3);

        var e3 = Main.Memory.newEdge(
            v3.index() / Main.Memory.Width.POINT,
            v4.index() / Main.Memory.Width.POINT,
            v4.distance(v3),
            v3, v4);

        var e4 = Main.Memory.newEdge(
            v4.index() / Main.Memory.Width.POINT,
            v1.index() / Main.Memory.Width.POINT,
            v1.distance(v4),
            v4, v1);

        // corner braces
        var e5 = Main.Memory.newEdge(
            v1.index() / Main.Memory.Width.POINT,
            v3.index() / Main.Memory.Width.POINT,
            v3.distance(v1),
            v1, v3);

        var e6 = Main.Memory.newEdge(
            v2.index() / Main.Memory.Width.POINT,
            v4.index() / Main.Memory.Width.POINT,
            v4.distance(v2),
            v2, v4);

        var edges = new FEdge2D[]{ e1, e2, e3, e4, e5, e6 };

        var force = 500;

        MathEX.centroid(verts, vectorBuffer);

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            0,0,0,0,
            v1.index() / Main.Memory.Width.POINT,
            v4.index() / Main.Memory.Width.POINT,
            e1.index() / Main.Memory.Width.EDGE,
            e6.index() / Main.Memory.Width.EDGE,
            verts, edges, force, entity);
    }
}
