package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.FBounds2D;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
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
    Map<BoxKey, Set<Integer>> boxMap = new ConcurrentHashMap<>();
    int[] keyDirectory = new int[xsubdivisions * ysubdivisions];

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

    public int[] rebuildIndex()
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

        for (Map.Entry<BoxKey, Set<Integer>> entry : boxMap.entrySet())
        {
            BoxKey key = entry.getKey();
            Set<Integer> value = entry.getValue();
            int[] matches = value.stream().mapToInt(i->i).toArray();
            int index = Main.Memory.storeKeyPointer(matches);
            int i = xsubdivisions * key.y + key.x;
            keyDirectory[i] = index;
        }
        return keyDirectory;
    }

    private static boolean doBoxesIntersect(FBounds2D a, FBounds2D b)
    {
        return a.x() < b.x() + b.w()
            && a.x() + a.w() > b.x()
            && a.y() < b.y() + b.h()
            && a.y() + a.h() > b.y();
    }

    private int[] matches(int[] keys, int[] key_directory)
    {
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
                int next = Main.Memory.pointer_buffer[j];
                rSet.add(next);
            }
        }
        int[] out = rSet.stream().mapToInt(i->i).toArray();
        return out;
    }

    private void doX(int location, int[] key_directory)
    {
        int bodyOffset = location * Main.Memory.Width.BODY;
        int bodyIndex = bodyOffset / Main.Memory.Width.BODY;

        var target = Main.Memory.bodyByIndex(bodyIndex);
        var bounds = target.bounds();
        var spatial_index = (int) bounds.si_index();
        var spatial_length = (int) bounds.si_length();

        var keys = new int[spatial_length];
        System.arraycopy(Main.Memory.key_buffer, spatial_index, keys, 0, spatial_length);

        var m = matches(keys, key_directory);
        //var c = getMatches(bodyIndex);

        for (int candidateIndex : m)
        {
            var candidate = Main.Memory.bodyByIndex(candidateIndex);
            if (target == candidate)
            {
                continue;
            }
            if (target.index() > candidate.index())
            {
                continue;
            }
            if (target.entity().equals(candidate.entity()))
            {
                continue;
            }
            boolean ch = doBoxesIntersect(target.bounds(), candidate.bounds());
            if (!ch)
            {
                continue;
            }

            //synchronized (outBuffer)
            //{
                outBuffer.add(bodyIndex);
                outBuffer.add(candidateIndex);
            //}
        }
    }


    //private final Map<String, Set<String>> collisionProgress = new HashMap<>();


    private final List<Integer> outBuffer = new ArrayList<>();

    public IntBuffer computeCandidates(int[] key_directory)
    {
        var bodyCount = Main.Memory.bodyCount();
        outBuffer.clear();
        //collisionProgress.clear();
        for (int location = 0; location < bodyCount; location++)
        {
            final int next = location;
            this.doX(next, key_directory);
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
