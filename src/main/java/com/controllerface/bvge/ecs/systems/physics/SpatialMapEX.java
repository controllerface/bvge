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

        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                var bodyKey = getKeyByIndex(current_x, current_y);
                // todo: remove this map and use a raw array
                boxMap.computeIfAbsent(bodyKey, SpatialMapEX::newKeySet).add(bodyIndex);
            }
        }
    }

    /**
     * Generates the keys for the given body, stores them in the appropriate section of the key bank,
     * and returns the total number of keys added. The provided map count array is also incremented
     * in the appropriate index in order to keep track of the total number of bodies that have a key
     * at that index.
     *
     * @param bodyIndex index of the body to generate keys for
     * @param keyBankBuffer the key bank to store the keys within
     * @param keyCounts the running counts array of keys at a given index
     * @return
     */
    private void generateBodyKeys(int bodyIndex, int[] keyBankBuffer, int[] keyCounts)
    {
        var body = Main.Memory.bodyByIndex(bodyIndex);

        var offset = body.bounds().bank_offset() * Main.Memory.Width.KEY;

        int min_x = body.si_min_x();
        int max_x = body.si_max_x();
        int min_y = body.si_min_y();
        int max_y = body.si_max_y();

        int current_index = offset;
        int totalCount = 0;

        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                keyBankBuffer[current_index++] = current_x;
                keyBankBuffer[current_index++] = current_y;
                int i = xsubdivisions * current_y + current_x;
                // todo: can a sum/reduction in CL work here to calculate total counts
                //  in parallel and then store in the map counts by index in the final step?
                keyCounts[i] = totalCount++; // increment the map count for this key
            }
        }
    }

    public int[] keyDirectory()
    {
        return keyDirectory;
    }

    public void updateKeyDirectory(int[] keyMap)
    {
        var t = 0;
        for (Map.Entry<BoxKey, Set<Integer>> entry : boxMap.entrySet())
        {
            // here, the 2D grid of the spatial map is packed into a 1D array
            // so it can be used in OpenCL.
            BoxKey key = entry.getKey();
            Set<Integer> value = entry.getValue();
            if (value.size() < 1) continue; // no matches in this key index
            int[] matches = value.stream().mapToInt(i->i).toArray();
            //System.arraycopy(matches, 0, keyMap, t, matches.length);
            int index = Main.Memory.storeKeyPointer(matches);
            int i = xsubdivisions * key.y + key.x;
            keyDirectory[i] = index;
            t += matches.length;
        }
        System.out.println("debug: " + t);
    }


    public int calculateKeyBankSize()
    {
        int size = 0;
        for (int bodyindex = 0; bodyindex < Main.Memory.bodyCount(); bodyindex++)
        {
            var body = Main.Memory.bodyByIndex(bodyindex);
            // write the current offset into the bounds object, this is the offset
            // into the main key bank where the keys for this body are stored.
            // when accessing keys, the si_bank size is used along with this
            // value to get all the keys for this object
            // todo: can a "scan" (parallel prefix sum) be used here? bodies could have their offset
            //  computed and set in CL this way
            body.bounds().setBankOffset(size / Main.Memory.Width.KEY);
            size += body.si_bank_size();
        }
        return size;
    }

    public void rebuildKeyBank(int[] keyBankBuffer, int[] keyCounts)
    {
        var bodyCount = Main.Memory.bodyCount();
        for (int bodyIndex = 0; bodyIndex < bodyCount; bodyIndex++)
        {
            generateBodyKeys(bodyIndex, keyBankBuffer, keyCounts);
        }
    }

    public void calculateMapOffsets(int[] keyOffsets, int[] keyCounts)
    {
        int currentOffset = 0;
        for (int i = 0; i < keyCounts.length; i++)
        {
            int next = keyCounts[i];
            if (next == 0)
            {
                keyOffsets[i] =-1;
                continue;
            }
            keyOffsets[i] = Integer.valueOf(currentOffset);
            currentOffset += next;
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
            rebuildLocation(location);
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
     * For a set of keys, belonging to an object, find all the other objects that share at least
     * one key.
     * @param keys keys for the target object
     * @param key_directory directory of potential matches
     * @return indices of all matching bodies
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

    private void findCandidates(int targetIndex, int[] key_directory, int[] key_bank)
    {
        var target = Main.Memory.bodyByIndex(targetIndex);
        var bounds = target.bounds();
        var spatial_index = bounds.bank_offset() * Main.Memory.Width.KEY;
        var spatial_length = target.si_bank_size();

        var keys = new int[spatial_length];
        System.arraycopy(key_bank, spatial_index, keys, 0, spatial_length);

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

    public IntBuffer computeCandidates(int[] key_directory, int[] key_bank)
    {
        var bodyCount = Main.Memory.bodyCount();
        outBuffer.clear();
        for (int targetIndex = 0; targetIndex < bodyCount; targetIndex++)
        {
            this.findCandidates(targetIndex, key_directory, key_bank);
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
        var y_map = keyMap.computeIfAbsent(index_x, SpatialMapEX::newKeyMap);
        return y_map.computeIfAbsent(index_y, (_k) -> new BoxKey(index_x, index_y));
    }
}
