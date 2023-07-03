package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.FBounds2D;

import java.nio.IntBuffer;
import java.util.*;

public class SpatialMap
{
    private float width = 1920;
    private float height = 1080;
    private int x_subdivisions = 200;
    private int y_subdivisions = 200;

    private int directoryLength;
    private float x_spacing = 0;
    private float y_spacing = 0;

    public SpatialMap()
    {
        init();
    }

    // todo: partitioning needs to change a bit, instead of specifying subdivisions, the spacing is what
    //  should be static, so the resize operation will keep the space cell size but make the tracking area
    //  alone bigger or smaller. right now, resize changes the cell size which is not ideal.
    void init()
    {
        x_spacing = width / x_subdivisions;
        y_spacing = height / y_subdivisions;
        directoryLength = x_subdivisions * y_subdivisions;
    }

    public void resize(float width, float height)
    {
        this.width = width;
        this.height = height;
        init();
    }

    private void rebuildLocationEX(int body_index, int[] key_map, int[] key_counts, int[] key_offsets)
    {
        var body = Main.Memory.bodyByIndex(body_index);

        int min_x = body.si_min_x();
        int max_x = body.si_max_x();
        int min_y = body.si_min_y();
        int max_y = body.si_max_y();

        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                int key_index = x_subdivisions * current_y + current_x;
                if (key_index < 0 || key_index >= key_counts.length)
                {
                    continue;
                }
                int count = key_counts[key_index];
                int offset = key_offsets[key_index];
                for (int i = offset; i < offset + count; i++)
                {
                    // todo: this could be reworked using a counter array
                    if (key_map[i] == -1)
                    {
                        key_map[i] = body_index;
                        break;
                    }
                }
            }
        }
    }

    /**
     * Generates the keys for the given body, stores them in the appropriate section of the key bank,
     * and returns the total number of keys added. The provided map count array is also incremented
     * in the appropriate index in order to keep track of the total number of bodies that have a key
     * at that index.
     *
     * @param body_index index of the body to generate keys for
     * @param key_bank the key bank to store the keys within
     * @param key_counts the running counts array of keys at a given index
     * @return
     */
    private void generateBodyKeys(int body_index, int[] key_bank, int[] key_counts)
    {
        var body = Main.Memory.bodyByIndex(body_index);

        var offset = body.bounds().bank_offset() * Main.Memory.Width.KEY;

        int min_x = body.si_min_x();
        int max_x = body.si_max_x();
        int min_y = body.si_min_y();
        int max_y = body.si_max_y();

        int current_index = offset;
        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                int key_index = x_subdivisions * current_y + current_x;
                if (key_index < 0 || key_index >= key_counts.length)
                {
                    continue;
                }

                key_bank[current_index++] = current_x;
                key_bank[current_index++] = current_y;

                // todo: can a sum/reduction in CL work here to calculate total counts
                //  in parallel and then store in the map counts by index in the final step?
                //  or perhaps atomic counters
                key_counts[key_index]++; // increment the map count for this key
            }
        }
    }

    public int directoryLength()
    {
        return directoryLength;
    }

    public int calculateKeyBankSize()
    {
        int size = 0;
        for (int body_index = 0; body_index < Main.Memory.bodyCount(); body_index++)
        {
            var body = Main.Memory.bodyByIndex(body_index);
            // write the current offset into the bounds object, this is the offset
            // into the main key bank where the keys for this body are stored.
            // when accessing keys, the si_bank size is used along with this
            // value to get all the keys for this object
            // todo: can a "scan" (parallel prefix sum) be used here? bodies could have their offset
            //  computed and set in CL this way
            body.bounds().setBankOffset(size / Main.Memory.Width.KEY);

            // todo: can this be calculated alone using a parallel reduce? maybe first?
            size += body.si_bank_size();
        }
        return size;
    }

    public void buildKeyBank(int[] key_bank, int[] key_counts)
    {
        var body_count = Main.Memory.bodyCount();
        for (int body_index = 0; body_index < body_count; body_index++)
        {
            generateBodyKeys(body_index, key_bank, key_counts);
        }
    }

    public void calculateMapOffsets(int[] key_offsets, int[] key_counts)
    {
        int current_offset = 0;
        for (int i = 0; i < key_counts.length; i++)
        {
            int next = key_counts[i];
            if (next == 0)
            {
                key_offsets[i] =-1;
                continue;
            }
            // todo: definitely would need a parallel prefix sum for this one, probably won't
            //  be able to use the -1 trick though, so may need to account for a difference between
            //  "no keys" at index vs. index offset _actually_ being 0.
            key_offsets[i] = current_offset;
            current_offset += next;
        }
    }

    public void buildKeyMap(int[] key_map, int[] key_counts, int[] key_offsets)
    {
        var body_count = Main.Memory.bodyCount();
        for (int location = 0; location < body_count; location++)
        {
            rebuildLocationEX(location, key_map, key_counts, key_offsets);
        }
    }

    private static boolean doBoxesIntersect(FBounds2D a, FBounds2D b)
    {
        return a.x() < b.x() + b.w()
            && a.x() + a.w() > b.x()
            && a.y() < b.y() + b.h()
            && a.y() + a.h() > b.y();
    }


    private int[] findMatchesEX(int target_index,
                                int[] keys,
                                int[] key_map,
                                int[] key_counts,
                                int[] key_offsets)
    {
        // use a set as a buffer for matches todo: can this size be pre-calculated?
        var rSet = new HashSet<Integer>();
        for (int i = 0; i < keys.length; i += Main.Memory.Width.KEY)
        {
            int x = keys[i];
            int y = keys[i + 1];
            int key_index = x_subdivisions * y + x;

            int count = key_counts[key_index];
            int offset = key_offsets[key_index];

            // these are sentinel values marking a key as having no entries
            if (count == 0 || offset == -1)
            {
                continue;
            }
            int[] hits = new int[count];
            System.arraycopy(key_map, offset, hits, 0, count);

            var target = Main.Memory.bodyByIndex(target_index);
            for (int j = 0; j < hits.length;j++)
            {
                int next = hits[j];
                if (target_index >= next)
                {
                    continue;
                }
                var candidate = Main.Memory.bodyByIndex(next);
                boolean ch = doBoxesIntersect(target.bounds(), candidate.bounds());
                if (!ch)
                {
                    continue;
                }
                rSet.add(next);
            }
        }
        // dump the buffer to an array todo: see above, can this be pre-sized?
        return rSet.stream().mapToInt(i->i).toArray();
    }

    private int[] findCandidatesEX(int target_index,
                                   int[] key_bank,
                                   int[] key_map,
                                   int[] key_counts,
                                   int[] key_offsets)
    {
        var target = Main.Memory.bodyByIndex(target_index);
        var bounds = target.bounds();
        var spatial_index = bounds.bank_offset() * Main.Memory.Width.KEY;
        var spatial_length = target.si_bank_size();
        var keys = new int[spatial_length];
        System.arraycopy(key_bank, spatial_index, keys, 0, spatial_length);
        return findMatchesEX(target_index, keys, key_map, key_counts, key_offsets);
    }

    // todo: factor this out to a statically sized array
    private final List<Integer> outBuffer = new ArrayList<>();

    public IntBuffer computeCandidatesEX(int[] key_bank,
                                         int[] key_map,
                                         int[] key_counts,
                                         int[] key_offsets)
    {
        var body_count = Main.Memory.bodyCount();
        outBuffer.clear();
        for (int target_index = 0; target_index < body_count; target_index++)
        {
            var matches = this.findCandidatesEX(target_index, key_bank, key_map, key_counts, key_offsets);
            for (int match : matches)
            {
                outBuffer.add(target_index);
                outBuffer.add(match);
            }
        }
        int[] ob = new int[outBuffer.size()];
        for (int i = 0; i < outBuffer.size(); i++)
        {
            ob[i] = outBuffer.get(i);
        }
        return IntBuffer.wrap(ob);
    }

    public float getWidth() {
        return width;
    }

    public float getHeight() {
        return height;
    }

    public float getX_spacing() {
        return x_spacing;
    }

    public float getY_spacing() {
        return y_spacing;
    }
}
