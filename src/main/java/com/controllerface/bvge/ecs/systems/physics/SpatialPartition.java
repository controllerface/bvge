package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.FBounds2D;

import java.nio.IntBuffer;
import java.util.*;

public class SpatialPartition
{
    private float width = 0;
    private float height = 0;

    // note: sub-divisions should always be divisible by 2
    private int x_subdivisions = 120;
    private int y_subdivisions = 120;

    private int directoryLength;
    private float x_spacing = 0;
    private float y_spacing = 0;

    private float x_origin = 0;
    private float y_origin = 0;

    public SpatialPartition()
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

    public void updateOrigin(float x_origin, float y_origin)
    {
        this.x_origin = x_origin;
        this.y_origin = y_origin;
    }

    private int calculateKeyIndex(int x, int y)
    {
        int key_index = x_subdivisions * y + x;
        return key_index;
    }

    private void rebuildLocationEX(int body_index)
    {
        var body = Main.Memory.bodyByIndex(body_index);

        if (!isInBounds(body.bounds()))
        {
            return;
        }

        int min_x = body.bounds().si_min_x();
        int max_x = body.bounds().si_max_x();
        int min_y = body.bounds().si_min_y();
        int max_y = body.bounds().si_max_y();

        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                int key_index = calculateKeyIndex(current_x, current_y);
                if (key_index < 0 || key_index >= key_counts.length)
                {
                    continue;
                }
                int count = key_counts[key_index];
                int offset = key_offsets[key_index];

                // this loop goes through the key map starting at the offset, up to the
                // count for this index, and finds an empty location in the precomputed
                // section of the map and places this body's index there. It doesn't
                // actually matter what the order is of indices being placed into the key
                // map, only that the position relative to the offset is only written to
                // once, so bodies don't overwrite other bodies' entries.
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

        boolean inBounds = isInBounds(body.bounds());

        if (!inBounds)
        {
            return;
        }

        var offset = body.bounds().bank_offset() * Main.Memory.Width.KEY;

        int min_x = body.bounds().si_min_x();
        int max_x = body.bounds().si_max_x();
        int min_y = body.bounds().si_min_y();
        int max_y = body.bounds().si_max_y();

        int current_index = offset;
        for (int current_x = min_x; current_x <= max_x; current_x++)
        {
            for (int current_y = min_y; current_y <= max_y; current_y++)
            {
                int key_index = calculateKeyIndex(current_x, current_y);
                if (key_index < 0 || current_index < 0
                    || key_index >= key_counts.length
                    || current_index >= key_bank.length)
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


    int key_bank_size = 0;
    int key_map_size = 0;

    int[] key_bank = new int[0];
    int[] key_map = new int[0];

    int[] key_counts = new int[0];
    int[] key_offsets = new int[0];

    public int calculateKeyBankSize()
    {
        int size = 0;
        for (int body_index = 0; body_index < Main.Memory.bodyCount(); body_index++)
        {
            var body = Main.Memory.bodyByIndex(body_index);
//            if (!isInBounds(body.bounds()))
//            {
//                body.bounds().setBankOffset(-1);
//            }
            // note: to keep logic simpler, objects that are out of bounds for this frame
            // are not filtered out. however, their bank sizes are set to zero during the
            // integration step, so they will not have an effect on the total size

            // write the current offset into the bounds object, this is the offset
            // into the main key bank where the keys for this body are stored.
            // when accessing keys, the si_bank size is used along with this
            // value to get all the keys for this object
            // todo: can a "scan" (parallel prefix sum) be used here? bodies could have their offset
            //  computed and set in CL this way
            //body.bounds().setBankOffset(size / Main.Memory.Width.KEY);

            //System.out.println(body_index + " : " + body.bounds().si_bank_size());
            // todo: can this be calculated alone using a parallel reduce? maybe first?
            size += body.bounds().si_bank_size();
        }

        key_bank_size = size;
        key_map_size = key_bank_size;// / Main.Memory.Width.KEY;
        // not sure why, but the key map size needs to be equal to the bank size when the body count is
        // high enough. todo: check if this is a bug or not?

        key_bank = new int[key_bank_size];
        key_map = new int[key_map_size];
        // todo: this -1 thing is a bit hacky, but needed for the moment to ensure the key map is built
        //  correctly. an alternative would be to make body index 0 unused, so indices all start at one,
        //  but that may create a lot more issues.
        Arrays.fill(key_map, -1);
        key_counts=new int[directoryLength];
        key_offsets=new int[directoryLength];
        return size;
    }

    public void buildKeyBank()
    {
        var body_count = Main.Memory.bodyCount();
        for (int body_index = 0; body_index < body_count; body_index++)
        {
            generateBodyKeys(body_index, key_bank, key_counts);
        }
    }

    public void calculateMapOffsets()
    {
        // todo: definitely would need a parallel prefix sum for this one
        int current_offset = 0;
        for (int i = 0; i < key_counts.length; i++)
        {
            int next = key_counts[i];
            key_offsets[i] = current_offset;
            current_offset += next;
        }
    }

    public void buildKeyMap()
    {
        var body_count = Main.Memory.bodyCount();
        for (int location = 0; location < body_count; location++)
        {
            rebuildLocationEX(location);
        }
    }

    private static boolean doBoxesIntersect(FBounds2D a, FBounds2D b)
    {
        return a.x() < b.x() + b.w()
            && a.x() + a.w() > b.x()
            && a.y() < b.y() + b.h()
            && a.y() + a.h() > b.y();
    }

    public boolean isInBounds(FBounds2D a)
    {
        return a.x() < x_origin + width
                && a.x() + a.w() > x_origin
                && a.y() < y_origin + height
                && a.y() + a.h() > y_origin;
    }


    private int[] findMatchesEX(int target_index,
                                int[] target_keys)
    {
        // use a set as a buffer for matches todo: can this size be pre-calculated?
        var rSet = new HashSet<Integer>();
        for (int i = 0; i < target_keys.length; i += Main.Memory.Width.KEY)
        {
            int x = target_keys[i];
            int y = target_keys[i + 1];
            int key_index = calculateKeyIndex(x, y);

            int count = key_counts[key_index];
            int offset = key_offsets[key_index];

            // these are sentinel values marking a key as having no entries
            if (count == 0)// || offset == -1)
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

    private int[] findCandidatesEX(int target_index)
    {
        var target = Main.Memory.bodyByIndex(target_index);
        var bounds = target.bounds();
        if (!isInBounds(bounds))
        {
            return new int[0];
        }
        var spatial_index = bounds.bank_offset() * Main.Memory.Width.KEY;
        var spatial_length = target.bounds().si_bank_size();
        var target_keys = new int[spatial_length];
        System.arraycopy(key_bank, spatial_index, target_keys, 0, spatial_length);
        return findMatchesEX(target_index, target_keys);
    }

    // todo: factor this out to a statically sized array
    private final List<Integer> outBuffer = new ArrayList<>();

    public IntBuffer computeCandidatesEX()
    {
        var body_count = Main.Memory.bodyCount();
        outBuffer.clear();
        for (int target_index = 0; target_index < body_count; target_index++)
        {
            var matches = this.findCandidatesEX(target_index);
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

    public float getX_origin() {
        return x_origin;
    }

    public float getY_origin() {
        return y_origin;
    }

    public int getX_subdivisions() {
        return x_subdivisions;
    }

    public int getY_subdivisions() {
        return y_subdivisions;
    }

    int[] getKeyForPoint(float px, float py)
    {
        float adjusted_x = px - (x_origin);
        float adjusted_y = py - (y_origin);
        int index_x = ((int) Math.floor(adjusted_x / x_spacing));
        int index_y = ((int) Math.floor(adjusted_y / y_spacing));
        int[] out = new int[2];
        out[0] = index_x;
        out[1] = index_y;
        return out;
    }

    public int countAtIndex(float x, float y)
    {
        if (x < x_origin || x > x_origin + width)
        {
            return 0;
        }

        int[] key = getKeyForPoint(x, y);
        int i = calculateKeyIndex(key[0], key[1]);
        if (i > key_counts.length-1)
        {
            return 0;
        }
        return key_counts[i];
    }
}
