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
    Map<Integer, Map<Integer, BoxKey>> keyMap = new HashMap<>();
    Map<BoxKey, Set<Integer>> boxMap = new HashMap<>();
    Map<Integer, Set<BoxKey>> bodyKeys = new HashMap<>();

    public SpatialMapEX()
    {
        init();
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
    }

    public void init()
    {
        x_spacing = width / xsubdivisions;
        y_spacing = height / ysubdivisions;
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

    private BoxKey getKeyByIndex(int index_x, int index_y)
    {
        return keyMap.computeIfAbsent(index_x, _k -> new HashMap<>())
            .computeIfAbsent(index_y, _k -> new BoxKey(index_x, index_y));
    }
}
