package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.FBounds2D;

import java.nio.IntBuffer;
import java.util.*;
import java.util.concurrent.*;

public class SpatialMapEX
{
    private float width = 1920;
    private float height = 1080;
    private int xsubdivisions = 250;
    private int ysubdivisions = 250;
    private float x_spacing = 0;
    private float y_spacing = 0;
    Map<Integer, Map<Integer, BoxKey>> keyMap = new ConcurrentHashMap<>();

    // maps all box keys within the tracked area to a set of body IDs
    // that have that box key in their key bank
    Map<BoxKey, Set<Integer>> boxMap = new ConcurrentHashMap<>();

    // after the map is built, this directory is updated, which effectively
    // duplicates it in memory, making it queryable as a raw structure
    private final int[] keyDirectory = new int[xsubdivisions * ysubdivisions];

    public SpatialMapEX()
    {
        init();
    }

    private static Set<Integer> newKeySet(BoxKey _k)
    {
        return new HashSet<>();
    }

    private static Map<Integer, BoxKey> newKeyMap(Integer _k)
    {
        return new HashMap<>();
    }

    void init()
    {
        x_spacing = width / xsubdivisions;
        y_spacing = height / ysubdivisions;
    }

    private void rebuildLocation(int bodyIndex)
    {
        var body = Main.Memory.bodyByIndex(bodyIndex);

        int min_x = body.si_min_x();
        int max_x = body.si_max_x();
        int min_y = body.si_min_y();
        int max_y = body.si_max_y();

        int current_index = 0;
        int[] key_bank = new int[body.si_bank_size()];
        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                key_bank[current_index++] = current_x;
                key_bank[current_index++] = current_y;
                var bodyKey = getKeyByIndex(current_x, current_y);

                // todo: remove the class based components
                boxMap.computeIfAbsent(bodyKey, SpatialMapEX::newKeySet).add(bodyIndex);
            }
        }
        var si_data = Main.Memory.storeKeyBank(key_bank);
        body.bounds().setSpatialIndex(si_data);
    }

    public int[] keyDirectory()
    {
        return keyDirectory;
    }

    public void updateKeyDirectory()
    {
        for (Map.Entry<BoxKey, Set<Integer>> entry : boxMap.entrySet())
        {
            // here, the 2D grid of the spatial map is packed into a 1D array
            // so it can be used in OpenCL.
            BoxKey key = entry.getKey();
            Set<Integer> value = entry.getValue();
            int[] matches = value.stream().mapToInt(i->i).toArray();
            int index = Main.Memory.storeKeyPointer(matches);
            int i = xsubdivisions * key.y + key.x;
            keyDirectory[i] = index;
        }
    }



    public void rebuildIndex()
    {
        keyMap.clear();
        boxMap.clear();
        Main.Memory.startKeyRebuild();
        var bodyCount = Main.Memory.bodyCount();
        for (int location = 0; location < bodyCount; location++)
        {
            int bodyOffset = location * Main.Memory.Width.BODY;
            int bodyIndex = bodyOffset / Main.Memory.Width.BODY;
            rebuildLocation(bodyIndex);
        }
    }

    private static boolean doBoxesIntersect(FBounds2D a, FBounds2D b)
    {
        return a.x() < b.x() + b.w()
            && a.x() + a.w() > b.x()
            && a.y() < b.y() + b.h()
            && a.y() + a.h() > b.y();
    }

    /**
     * For a set of keys, belonging to an object, find all of the other objects that share at least
     * one key.
     * @param keys keys for the target object
     * @param key_directory directory of potential matches
     * @return
     */
    private int[] findMatches(int target, int[] keys, int[] key_directory)
    {
        // use a set as a buffer for matches todo: can this size be pre-calculated?
        var rSet = new HashSet<Integer>();
        for (int i = 0; i < keys.length; i += Main.Memory.Width.KEY)
        {
            int x = keys[i];
            int y = keys[i + 1];
            int pointer = key_directory[xsubdivisions * y + x];
            int len = Main.Memory.pointer_buffer[pointer];
            int endIndex = pointer + len;
            for (int j = pointer + 1; j <= endIndex; j++)
            {
                // get the next potential match
                int next = Main.Memory.pointer_buffer[j];
                // if this would be a self collision, or a duplicate to an earlier match,
                // discard this match.
                if (target >= next)
                {
                    continue;
                }
                rSet.add(next);
            }
        }
        // dump the buffer to an array todo: see above, can this be pre-sized?
        return rSet.stream().mapToInt(i->i).toArray();
    }

    private final List<Integer> outBuffer = new ArrayList<>();

    private void findCandidates(int targetIndex, int[] key_directory)
    {
        var target = Main.Memory.bodyByIndex(targetIndex);
        var bounds = target.bounds();
        var spatial_index = (int) bounds.si_index();
        var spatial_length = (int) bounds.si_length();

        var keys = new int[spatial_length];
        System.arraycopy(Main.Memory.key_buffer, spatial_index, keys, 0, spatial_length);

        var matches = findMatches(targetIndex, keys, key_directory);

        for (int candidateIndex : matches)
        {
            var candidate = Main.Memory.bodyByIndex(candidateIndex);
            boolean ch = doBoxesIntersect(target.bounds(), candidate.bounds());
            if (!ch)
            {
                continue;
            }
            outBuffer.add(targetIndex);
            outBuffer.add(candidateIndex);
        }
    }

    public IntBuffer computeCandidates(int[] key_directory)
    {
        var bodyCount = Main.Memory.bodyCount();
        outBuffer.clear();
        for (int location = 0; location < bodyCount; location++)
        {
            this.findCandidates(location, key_directory);
        }
        int[] ob = new int[outBuffer.size()];
        for (int i = 0; i < outBuffer.size(); i++)
        {
            ob[i] = outBuffer.get(i);
        }
        return IntBuffer.wrap(ob);
    }

    private BoxKey getKeyByIndex(int index_x, int index_y)
    {
        var one = keyMap.computeIfAbsent(index_x, SpatialMapEX::newKeyMap);
        var two = one.computeIfAbsent(index_y, (_k) -> new BoxKey(index_x, index_y));
        return two;
    }
}
