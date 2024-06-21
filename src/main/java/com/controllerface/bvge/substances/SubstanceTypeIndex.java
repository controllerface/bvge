package com.controllerface.bvge.substances;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SubstanceTypeIndex
{
    private static final List<Enum<?>> offsets = new ArrayList<>();

    static
    {
        register_substance(Solid.class);
        register_substance(Liquid.class);
    }

    // TODO: expand this class out to manage yields

    public static <E extends Enum<E>> void register_substance(Class<E> enumClass)
    {
        Collections.addAll(offsets, enumClass.getEnumConstants());
    }

    public static int to_type_index(Enum<?> e)
    {
        return offsets.contains(e)
            ? offsets.indexOf(e)
            : -1;
    }

    public static Enum<?> from_type_index(int i)
    {
        if (i < 0) return null;
        if (i >= offsets.size()) return null;
        return offsets.get(i);
    }
}
