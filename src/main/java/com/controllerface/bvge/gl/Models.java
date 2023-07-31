package com.controllerface.bvge.gl;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Models
{
    private static final Map<Integer, float[]> loaded_models = new HashMap<>();
    private static final Map<Integer, Boolean> dirty_models = new HashMap<>();
    private static final Map<Integer, Set<Integer>> model_instances = new HashMap<>();

    public static float[] get_model_by_index(int index)
    {
        return loaded_models.get(index);
    }

    public static void register_model_instance(int model_id, int body_id)
    {
        model_instances.computeIfAbsent(model_id, _k->new HashSet<>())
            .add(body_id);
        dirty_models.put(model_id, true);
    }

    public static Set<Integer> get_model_instances(int model_id)
    {
        return model_instances.get(model_id);
    }

    public static boolean is_model_dirty(int model_id)
    {
        var r = dirty_models.get(model_id);
        return r != null && r;
    }

    public static int get_instance_count(int model_id)
    {
        return model_instances.get(model_id).size();
    }

    public static void set_model_clean(int model_id)
    {
        dirty_models.put(model_id, false);
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
