package com.controllerface.bvge.data;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.util.MathEX;
import org.joml.Vector2f;

/**
 * This is the core "factor" class for all physics based objects. It contains named archetype
 * methods that can be used to create tracked objects.
 */
public class PhysicsObjects
{
    private static final Vector2f vectorBuffer = new Vector2f();

    public static int FLAG_NONE = 0x00;
    public static int FLAG_STATIC = 0x01;

    public static FBody2D simpleBox(float x, float y, float size, String entity)
    {
        var halfSize = size / 2;

        var p1 = Main.Memory.newPoint(x - halfSize, y - halfSize);
        var p2 = Main.Memory.newPoint(x + halfSize, y - halfSize);
        var p3 = Main.Memory.newPoint(x + halfSize, y + halfSize);
        var p4 = Main.Memory.newPoint(x - halfSize, y + halfSize);

        var points = new FPoint2D[]{ p1, p2, p3, p4 };
        MathEX.centroid(points, vectorBuffer);

        // box sides
        var start_edge = Main.Memory.newEdge(
            p1.index() / Main.Memory.Width.POINT,
            p2.index() / Main.Memory.Width.POINT,
            p2.distance(p1),
            p1, p2);

        Main.Memory.newEdge(
            p2.index() / Main.Memory.Width.POINT,
            p3.index() / Main.Memory.Width.POINT,
            p3.distance(p2),
            p2, p3);

        Main.Memory.newEdge(
            p3.index() / Main.Memory.Width.POINT,
            p4.index() / Main.Memory.Width.POINT,
            p4.distance(p3),
            p3, p4);

        Main.Memory.newEdge(
            p4.index() / Main.Memory.Width.POINT,
            p1.index() / Main.Memory.Width.POINT,
            p1.distance(p4),
            p4, p1);

        // corner braces
        Main.Memory.newEdge(
            p1.index() / Main.Memory.Width.POINT,
            p3.index() / Main.Memory.Width.POINT,
            p3.distance(p1),
            p1, p3);

        var end_edge = Main.Memory.newEdge(
            p2.index() / Main.Memory.Width.POINT,
            p4.index() / Main.Memory.Width.POINT,
            p4.distance(p2),
            p2, p4);

        var force = 500;

        var bounds = Main.Memory.newBounds();

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            (float) p1.index() / Main.Memory.Width.POINT,
            (float) p4.index() / Main.Memory.Width.POINT,
            (float) start_edge.index() / Main.Memory.Width.EDGE,
            (float) end_edge.index() / Main.Memory.Width.EDGE,
            FLAG_NONE,
            bounds, force,
            entity);
    }

    public static FBody2D staticBox(float x, float y, float size, String entity)
    {
        var halfSize = size / 2;

        var p1 = Main.Memory.newPoint(x - halfSize, y - halfSize);
        var p2 = Main.Memory.newPoint(x + halfSize, y - halfSize);
        var p3 = Main.Memory.newPoint(x + halfSize, y + halfSize);
        var p4 = Main.Memory.newPoint(x - halfSize, y + halfSize);

        var points = new FPoint2D[]{ p1, p2, p3, p4 };
        MathEX.centroid(points, vectorBuffer);

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

        var force = 500;

        var bounds = Main.Memory.newBounds();

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            (float) p1.index() / Main.Memory.Width.POINT,
            (float) p4.index() / Main.Memory.Width.POINT,
            (float) e1.index() / Main.Memory.Width.EDGE,
            (float) e6.index() / Main.Memory.Width.EDGE,
            FLAG_STATIC,
            bounds, force,
            entity);
    }

    public static FBody2D polygon1(float x, float y, float size, String entity)
    {
        var halfSize = size / 2;

        var p1 = Main.Memory.newPoint(x - halfSize, y - halfSize);
        var p2 = Main.Memory.newPoint(x + halfSize, y - halfSize);
        var p3 = Main.Memory.newPoint(x + halfSize, y + halfSize);
        var p4 = Main.Memory.newPoint(x - halfSize, y + halfSize);
        var p5 = Main.Memory.newPoint(x, y + halfSize * 2);

        var points = new FPoint2D[]{ p1, p2, p3, p4, p5 };
        MathEX.centroid(points, vectorBuffer);

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


        var e5a = Main.Memory.newEdge(
                p4.index() / Main.Memory.Width.POINT,
                p5.index() / Main.Memory.Width.POINT,
                p4.distance(p5),
                p4, p5);

        var e5b = Main.Memory.newEdge(
                p3.index() / Main.Memory.Width.POINT,
                p5.index() / Main.Memory.Width.POINT,
                p3.distance(p5),
                p3, p5);

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


        var force = 2500;

        var bounds = Main.Memory.newBounds();

        return Main.Memory.newBody(vectorBuffer.x, vectorBuffer.y,
            size, size,
            0,0,
            (float) p1.index() / Main.Memory.Width.POINT,
            (float) p5.index() / Main.Memory.Width.POINT,
            (float) e1.index() / Main.Memory.Width.EDGE,
            (float) e6.index() / Main.Memory.Width.EDGE,
            FLAG_NONE,
            bounds, force,
            entity);
    }
}
