package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.ecs.RigidBody2D;
import com.controllerface.bvge.util.quadtree.QuadRectangle;

import java.util.*;

public class SpatialMap
{
    private final VerletPhysics verletPhysics;
    private float width = 1280;
    private float height = 628;
    private float xsubdivisions = 1;
    private float ysubdivisions = 1;
    private float x_spacing = 0;
    private float y_spacing = 0;
    public List<QuadRectangle> rects = new ArrayList<>();
    private Set<VerletPhysics.BoxKey> playerkeys = new HashSet<>();

    Map<Integer, Map<Integer, VerletPhysics.BoxKey>> keyMap = new HashMap<>();
    Map<VerletPhysics.BoxKey, Set<RigidBody2D>> boxMap = new HashMap<>();

    public SpatialMap(VerletPhysics verletPhysics)
    {
        this.verletPhysics = verletPhysics;
        init();
    }

    public void clear()
    {
        boxMap.values().forEach(Set::clear);
        playerkeys.clear();
    }

    public void rebuildRects()
    {
        rects.clear();
        var currentX = 0;
        var currentY = 0;
        for (int i = 0; i < xsubdivisions; i++)
        {
            for (int j = 0; j < ysubdivisions; j++)
            {
                var k = keyMap.get(i).get(j);
                if (playerkeys.contains(k))
                {
                    rects.add(new QuadRectangle(currentX, currentY, x_spacing, y_spacing, true));
                }
                else
                {
                    rects.add(new QuadRectangle(currentX, currentY, x_spacing, y_spacing));
                }
                currentY += y_spacing;
            }
            currentX += x_spacing;
            currentY = 0;
        }
    }

    public void init()
    {
        x_spacing = (float) width / (float) xsubdivisions;
        y_spacing = (float) height / (float) ysubdivisions;

        float currentX = 0;
        float currentY = 0;
        for (int i = 0; i < xsubdivisions; i++)
        {
            for (int j = 0; j < ysubdivisions; j++)
            {

                var k = new VerletPhysics.BoxKey(i, j);
                keyMap.computeIfAbsent(i, (_i) -> new HashMap<>()).put(j, k);
                boxMap.put(k, new HashSet<>());
                rects.add(new QuadRectangle(currentX, currentY, x_spacing, y_spacing));

                currentY += y_spacing;
            }
            currentX += x_spacing;
            currentY = 0;
        }
    }

    public Set<RigidBody2D> getMatches(QuadRectangle box)
    {
        var rSet = new HashSet<RigidBody2D>();
        for (VerletPhysics.BoxKey k : box.getKeys())
        {
            rSet.addAll(boxMap.get(k));
        }
        return rSet;
    }

    public void add(RigidBody2D body, QuadRectangle box)
    {
        box.resetKeys();
//        var t = verletPhysics.ecs.getComponentFor(body.getEntitiy(), Component.Transform);
//        Transform transform = Component.Transform.coerce(t);
//        var k0 = getKeyForPoint(transform.position.x, transform.position.y);
        var k1 = getKeyForPoint(box.x, box.y);
        var k2 = getKeyForPoint(box.x + box.width, box.y);
        var k3 = getKeyForPoint(box.x + box.width, box.y + box.height);
        var k4 = getKeyForPoint(box.x, box.y + box.height);

        if (/*k0 == null
            &&*/ k1 == null
            && k2 == null
            && k3 == null
            && k4 == null)
        {
            return;
        }

        // is within only one cell
        if (/*k0 == k1 &&*/ k1 == k2 && k1 == k3 && k1 == k4)
        {
            boxMap.get(k1).add(body);
            box.addkey(k1);
            return;
        }

//        if (k0 != null)
//        {
//            boxMap.get(k0).add(body);
//            box.addkey(k0);
//        }

        var keys = new VerletPhysics.BoxKey[]{ k1, k2, k3, k4 };
        var min_x = Integer.MAX_VALUE;
        var max_x = Integer.MIN_VALUE;
        var min_y = Integer.MAX_VALUE;
        var max_y = Integer.MIN_VALUE;
        for (VerletPhysics.BoxKey k : keys)
        {
            if (k == null) continue;

            if (k.x > max_x)
            {
                max_x = k.x;
            }
            if (k.x < min_x)
            {
                min_x = k.x;
            }

            if (k.y > max_y)
            {
                max_y = k.y;
            }
            if (k.y < min_y)
            {
                min_y = k.y;
            }
        }
        boolean isPlayer = false;
        if (body.getEntitiy().equals("player"))
        {
            isPlayer = true;
//            playerkeys.addAll(keys);
//            rebuildRects();
        }
        for (int i = min_x; i <= max_x; i++)
        {
            for (int j = min_y; j <= max_y; j++)
            {
                var k = getKeyByIndex(i, j);
                if (k == null) continue;
                boxMap.get(k).add(body);
                box.addkey(k);
                if (isPlayer) playerkeys.add(k);
            }
        }
        if (isPlayer) rebuildRects();

//        if (k1 != null)
//        {
//            boxMap.get(k1).add(body);
//            box.addkey(k1);
//        }
//        if (k2 != null)
//        {
//            boxMap.get(k2).add(body);
//            box.addkey(k2);
//        }
//        if (k3 != null)
//        {
//            boxMap.get(k3).add(body);
//            box.addkey(k3);
//        }
//        if (k4 != null)
//        {
//            boxMap.get(k4).add(body);
//            box.addkey(k4);
//        }
//
//        if (body.getEntitiy().equals("player"))
//        {
//            playerkeys.addAll(keys);
//            rebuildRects();
//        }
    }


    private VerletPhysics.BoxKey getKeyByIndex(int index_x, int index_y)
    {
//            int index_x = px - (px % x_spacing);
//            int index_y = py - (py % y_spacing);

//        int index_x = ((int) Math.floor(px / x_spacing));
//        int index_y = ((int) Math.floor(py / y_spacing));
        if (!keyMap.containsKey(index_x))
        {
            return null;
        }
        var ymp = keyMap.get(index_x);
        if (!ymp.containsKey(index_y))
        {
            return null;
        }
        return ymp.get(index_y);
    }

    private VerletPhysics.BoxKey getKeyForPoint(float px, float py)
    {
//            int index_x = px - (px % x_spacing);
//            int index_y = py - (py % y_spacing);

        int index_x = ((int) Math.floor(px / x_spacing));
        int index_y = ((int) Math.floor(py / y_spacing));
        if (!keyMap.containsKey(index_x))
        {
            return null;
        }
        var ymp = keyMap.get(index_x);
        if (!ymp.containsKey(index_y))
        {
            return null;
        }
        return ymp.get(index_y);
    }
}
