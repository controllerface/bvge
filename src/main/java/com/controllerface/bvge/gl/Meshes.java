package com.controllerface.bvge.gl;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Meshes
{
    public static final int CIRCLE_MESH = 0;
    public static final int BOX_MESH = 1;
    public static final int POLYGON1_MESH = 2;

    private static final Map<Integer, float[]> loaded_meshes = new HashMap<>();
    private static final Map<Integer, Boolean> dirty_meshes = new HashMap<>();
    private static final Map<Integer, Set<Integer>> mesh_instances = new HashMap<>();

    public static float[] get_mesh_by_index(int index)
    {
        return loaded_meshes.get(index);
    }

    // todo: this may need to be changed so it points to a complete model, or possibly
    //  a central hull that may be one of several meshes contained in the model. Right now
    //  a model and mesh are 1:1 and that will need to change.
    public static void register_mesh_instance(int model_id, int hull_id)
    {
        mesh_instances.computeIfAbsent(model_id, _k -> new HashSet<>()).add(hull_id);
        dirty_meshes.put(model_id, true);
    }

    public static Set<Integer> get_mesh_instances(int model_id)
    {
        return mesh_instances.get(model_id);
    }

    public static boolean is_mesh_dirty(int model_id)
    {
        var r = dirty_meshes.get(model_id);
        return r != null && r;
    }

    public static int get_instance_count(int model_id)
    {
        return mesh_instances.get(model_id).size();
    }

    public static void set_mesh_clean(int model_id)
    {
        dirty_meshes.put(model_id, false);
    }

    /**
     * A simple unit square; 4 vertices defining a square of size 1.
     */
    private static float[] load_box_mesh()
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
    private static float[] load_poly1_mesh()
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
        loaded_meshes.put(CIRCLE_MESH, new float[]{ 0, 0 });
        loaded_meshes.put(BOX_MESH, load_box_mesh());
        loaded_meshes.put(POLYGON1_MESH, load_poly1_mesh());
    }
}
