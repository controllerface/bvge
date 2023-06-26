package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.ecs.components.QuadRectangle;

import java.util.*;

public class SpatialMapEX
{
    private float width = 1920;
    private float height = 1080;
    private float xsubdivisions = 250;
    private float ysubdivisions = 250;
    private float x_spacing = 0;
    private float y_spacing = 0;
    public List<QuadRectangle> rects = new ArrayList<>();
    private Set<BoxKey> playerkeys = new HashSet<>();

    Map<Integer, Map<Integer, BoxKey>> keyMap = new HashMap<>();
    Map<BoxKey, Set<Integer>> boxMap = new HashMap<>();
    Map<Integer, Set<BoxKey>> bodyKeys = new HashMap<>();

    public SpatialMapEX()
    {
        //init();
    }

    private static Set<BoxKey> newBoxSet(int _k)
    {
        return new HashSet<>();
    }

    private static Set<Integer> newIntSet(BoxKey _k)
    {
        return new HashSet<>();
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
        x_spacing = width / xsubdivisions;
        y_spacing = height / ysubdivisions;

        float currentX = 0;
        float currentY = 0;
        for (int i = 0; i < xsubdivisions; i++)
        {
            for (int j = 0; j < ysubdivisions; j++)
            {

                var k = new BoxKey(i, j);
                keyMap.computeIfAbsent(i, (_i) -> new HashMap<>()).put(j, k);
                boxMap.put(k, new HashSet<>());
                rects.add(new QuadRectangle(currentX, currentY, x_spacing, y_spacing));

                currentY += y_spacing;
            }
            currentX += x_spacing;
            currentY = 0;
        }
    }

    public void rebuildMatches()
    {
        boxMap.clear();
        bodyKeys.clear();
        // todo: this might work in an executor or fork/join pool
        for (int location = 0; location < Main.Memory.bodyCount(); location++)
        {
            int bodyOffset = location * Main.Memory.Width.BODY;
            int bodyIndex = bodyOffset / Main.Memory.Width.BODY;

            int min_x = (int)Main.Memory.body_buffer[bodyOffset + FBody2D.si_min_x_offset];
            int max_x = (int)Main.Memory.body_buffer[bodyOffset + FBody2D.si_max_x_offset];
            int min_y = (int)Main.Memory.body_buffer[bodyOffset + FBody2D.si_min_y_offset];
            int max_y = (int)Main.Memory.body_buffer[bodyOffset + FBody2D.si_max_y_offset];

            for (int i2 = min_x; i2 <= max_x; i2++)
            {
                for (int j2 = min_y; j2 <= max_y; j2++)
                {
                    var bodyKey = getKeyByIndex(i2, j2);
                    boxMap.computeIfAbsent(bodyKey, SpatialMapEX::newIntSet)
                        .add(bodyIndex);
                    bodyKeys.computeIfAbsent(bodyIndex, SpatialMapEX::newBoxSet)
                        .add(bodyKey);
                }
            }
        }
    }

    public Set<Integer> getMatches(Integer boxId)
    {
        var keys = bodyKeys.get(boxId);
        var rSet = new HashSet<Integer>();
        for (BoxKey k : keys)
        {
            rSet.addAll(boxMap.get(k));
        }
        return rSet;
    }

//    public void add(FBody2D body, QuadRectangle box)
//    {
//        box.resetKeys();
//
//        var k1 = getKeyForPoint(box.x, box.y);
//        var k2 = getKeyForPoint(box.x + box.width, box.y);
//        var k3 = getKeyForPoint(box.x + box.width, box.y + box.height);
//        var k4 = getKeyForPoint(box.x, box.y + box.height);
//
//        if (k1 == null
//            && k2 == null
//            && k3 == null
//            && k4 == null)
//        {
//            return;
//        }
//
//        // is within only one cell, so just add that key
//        if (k1 == k2 && k1 == k3 && k1 == k4)
//        {
//            boxMap.get(k1).add(body);
//            box.addkey(k1);
//            return;
//        }
//
//        // otherwise, we need p2 loop and get all of the keys that overlap this box
//        var keys = new BoxKey[]{k1, k2, k3, k4};
//        var min_x = Integer.MAX_VALUE;
//        var max_x = Integer.MIN_VALUE;
//        var min_y = Integer.MAX_VALUE;
//        var max_y = Integer.MIN_VALUE;
//        for (BoxKey k : keys)
//        {
//            if (k == null)
//            {
//                continue;
//            }
//
//            if (k.x > max_x)
//            {
//                max_x = k.x;
//            }
//            if (k.x < min_x)
//            {
//                min_x = k.x;
//            }
//
//            if (k.y > max_y)
//            {
//                max_y = k.y;
//            }
//            if (k.y < min_y)
//            {
//                min_y = k.y;
//            }
//        }
//        boolean isPlayer = false;
//        if (body.entity().equals("player"))
//        {
//            isPlayer = true;
//        }
//        for (int i = min_x; i <= max_x; i++)
//        {
//            for (int j = min_y; j <= max_y; j++)
//            {
//                var k = getKeyByIndex(i, j);
//                if (k == null)
//                {
//                    continue;
//                }
//                boxMap.get(k).add(body);
//                box.addkey(k);
//                if (isPlayer)
//                {
//                    playerkeys.add(k);
//                }
//            }
//        }
//        if (isPlayer)
//        {
//            rebuildRects();
//        }
//    }

    private BoxKey getKeyByIndex(int index_x, int index_y)
    {
        return keyMap.computeIfAbsent(index_x, _k -> new HashMap<>())
            .computeIfAbsent(index_y, _k -> new BoxKey(index_x, index_y));
    }

    private BoxKey getKeyForPoint(float px, float py)
    {
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
