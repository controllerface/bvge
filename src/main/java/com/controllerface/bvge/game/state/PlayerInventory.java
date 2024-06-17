package com.controllerface.bvge.game.state;

import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.substances.SubstanceTypeIndex;

import java.util.EnumMap;
import java.util.Map;

public class PlayerInventory
{
    private final Map<Solid, Integer> solid_counts = new EnumMap<>(Solid.class);

    public void collect_substance(int type_index, int quantity)
    {
        var type = SubstanceTypeIndex.from_type_index(type_index);
        switch (type)
        {
            case Solid s ->
            {
                System.out.println("collected solid: " + s);
                int current = solid_counts.computeIfAbsent(s, (_) -> 0);
                solid_counts.put(s, current + quantity);
            }
            case null -> throw new NullPointerException("Unknown type index: " + type_index);
            default   -> throw new IllegalStateException("Unexpected type: " + type);
        }
    }
}
