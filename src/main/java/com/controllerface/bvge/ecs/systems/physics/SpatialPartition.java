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

    int key_bank_size = 0;
    int key_map_size = 0;
    int[] key_bank = new int[0];
    int[] key_map = new int[0];
    int[] key_counts = new int[0];
    int[] key_offsets = new int[0];

    public void resizeBank(int size)
    {
        key_bank_size = size;
        key_map_size = size / 2;
        key_map     = new int[key_map_size];
        key_bank    = new int[key_bank_size];
        key_counts  = new int[directoryLength];
        key_offsets = new int[directoryLength];
    }

    public int[] getKey_counts()
    {
        return key_counts;
    }

    public int[] getKey_bank()
    {
        return key_bank;
    }

    public int[] getKey_offsets()
    {
        return key_offsets;
    }

    public int[] getKey_map()
    {
        return key_map;
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


    private int[] findMatchesEX(int target_index, int[] target_keys)
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

            // sentinel value marking a key as having no entries
            if (count == 0)
            {
                continue;
            }

            int[] hits = new int[count];
            System.arraycopy(key_map, offset, hits, 0, count);

            var target = Main.Memory.bodyByIndex(target_index);
            for (int j = 0; j < hits.length;j++)
            {
                int next = hits[j];
                // this is where duplicate/reverse collisions are weeded out
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
        if (bounds.si_bank_size() == 0)
        {
            return new int[0];
        }
        var spatial_index = bounds.bank_offset() * Main.Memory.Width.KEY;
        var spatial_length = bounds.si_bank_size();
        var target_keys = new int[spatial_length];
        // grab the keys for this object and use them to find potential candidates
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
