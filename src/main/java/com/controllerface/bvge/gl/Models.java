package com.controllerface.bvge.gl;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Models
{
    private static final Map<Integer, float[]> loaded_models = new HashMap<>();
    private static final Map<Integer, Set<Integer>> model_instances = new HashMap<>();

    public static float[] get_model_by_index(int index)
    {
        return loaded_models.get(index);
    }

    public static void register_model_instance(int model_id, int body_id)
    {
        model_instances.computeIfAbsent(model_id, _k->new HashSet<>())
            .add(body_id);
    }

    /**
     * A simple square with 4 vertices
     */
    private static float[] load_box_model()
    {
        float halfSize = 1f / 2f;
        float[] model = new float[8];
        model[0] = -halfSize;
        model[1] = -halfSize;
        model[2] = halfSize;
        model[3] = -halfSize;
        model[4] = halfSize;
        model[5] = halfSize;
        model[6] = -halfSize;
        model[7] = halfSize;
        return model;
    }

    /**
     * A simple polygon with 5 vertices
     */
    private static float[] load_poly1_model()
    {
        float halfSize = 1f / 2f;
        float[] model = new float[10];
        model[0] = -halfSize;
        model[1] = -halfSize;
        model[2] = halfSize;
        model[3] = -halfSize;
        model[4] = halfSize;
        model[5] = halfSize;
        model[6] = -halfSize;
        model[7] = halfSize;
        model[8] = 0;
        model[9] = halfSize * 2;
        return model;
    }

    public static void init()
    {
        loaded_models.put(0, load_box_model());
        loaded_models.put(1, load_poly1_model());
    }
}
