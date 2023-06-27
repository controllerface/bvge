package com.controllerface.bvge.ecs.systems.physics;

import com.controllerface.bvge.Main;
import com.controllerface.bvge.data.FBody2D;
import com.controllerface.bvge.data.FBounds2D;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class SpatialMapEX
{
    private float width = 1920;
    private float height = 1080;
    private float xsubdivisions = 250;
    private float ysubdivisions = 250;
    private float x_spacing = 0;
    private float y_spacing = 0;
    Map<Integer, Map<Integer, BoxKey>> keyMap = new ConcurrentHashMap<>();
    Map<BoxKey, Set<Integer>> boxMap = new ConcurrentHashMap<>();
    Map<Integer, Set<BoxKey>> bodyKeys = new ConcurrentHashMap<>();

    public SpatialMapEX()
    {
        init();
    }

    private static Set<BoxKey> newBoxSet(int _k)
    {
        return new HashSet<>();
    }

    private static Set<Integer> newKeySet(BoxKey _k)
    {
        return new HashSet<>();
    }

    private static Map<Integer, BoxKey> newKeyMap(Integer _k)
    {
        return new HashMap<>();
    }

    public void init()
    {
        x_spacing = width / xsubdivisions;
        y_spacing = height / ysubdivisions;
    }

    public void rebuildIndex()
    {
        keyMap.clear();
        boxMap.clear();
        bodyKeys.clear();
        var bodyCount = Main.Memory.bodyCount();
        var counter = new AtomicInteger(bodyCount);
        // todo: this might work in an executor or fork/join pool
        for (int location = 0; location < bodyCount; location++)
        {
            int bodyOffset = location * Main.Memory.Width.BODY;
            int bodyIndex = bodyOffset / Main.Memory.Width.BODY;
            var body = Main.Memory.bodyByIndex(bodyIndex);

            int min_x = body.si_min_x();
            int max_x = body.si_max_x();
            int min_y = body.si_min_y();
            int max_y = body.si_max_y();

            for (int currentX = min_x; currentX <= max_x; currentX++)
            {
                for (int currentY = min_y; currentY <= max_y; currentY++)
                {
                    var bodyKey = getKeyByIndex(currentX, currentY);
                    boxMap.computeIfAbsent(bodyKey, SpatialMapEX::newKeySet).add(bodyIndex);
                    bodyKeys.computeIfAbsent(bodyIndex, SpatialMapEX::newBoxSet).add(bodyKey);
                }
            }
            //System.out.println(counter.decrementAndGet());
        }
        //System.out.println("AT End" + counter.get());
    }

    private static byte[] intToBytes(final int data) {
        return new byte[] {
            (byte)((data >> 24) & 0xff),
            (byte)((data >> 16) & 0xff),
            (byte)((data >> 8) & 0xff),
            (byte)((data >> 0) & 0xff),
        };
    }

    private static boolean doBoxesIntersect(FBounds2D a, FBounds2D b)
    {
        return a.x() < b.x() + b.w()
            && a.x() + a.w() > b.x()
            && a.y() < b.y() + b.h()
            && a.y() + a.h() > b.y();
    }

    private void doX(int location)
    {
        int bodyOffset = location * Main.Memory.Width.BODY;
        int bodyIndex = bodyOffset / Main.Memory.Width.BODY;

        var target = Main.Memory.bodyByIndex(bodyIndex);
        var c = getMatches(bodyIndex);

        for (Integer candidateIndex : c)
        {
            var candidate = Main.Memory.bodyByIndex(candidateIndex);
            if (target == candidate)
            {
                continue;
            }
            if (target.entity().equals(candidate.entity()))
            {
                continue;
            }

            var keyA = "";
            var keyB = "";
            if (target.entity().compareTo(candidate.entity()) < 0)
            {
                keyA = target.entity();
                keyB = candidate.entity();
            }
            else
            {
                keyA = candidate.entity();
                keyB = target.entity();
            }

            synchronized (collisionProgress)
            {
                if (collisionProgress.computeIfAbsent(keyA, (k) -> new HashSet<>()).contains(keyB))
                {
                    continue;
                }
                collisionProgress.get(keyA).add(keyB);
            }

            boolean ch = doBoxesIntersect(target.bounds(), candidate.bounds());
            if (!ch)
            {
                continue;
            }

            synchronized (outBuffer)
            {
                outBuffer.add(bodyIndex);
                outBuffer.add(candidateIndex);
            }
        }
    }


    private final Map<String, Set<String>> collisionProgress = new HashMap<>();


    private final List<Integer> outBuffer = new ArrayList<>();

    public IntBuffer computeCandidates()
    {
        var bodyCount = Main.Memory.bodyCount();
        outBuffer.clear();
        collisionProgress.clear();
        for (int location = 0; location < bodyCount; location++)
        {
            final int next = location;
            this.doX(next);
        }
        int[] ob = new int[outBuffer.size()];
        for (int i = 0; i < outBuffer.size(); i++)
        {
            ob[i] = outBuffer.get(i);
        }
        return IntBuffer.wrap(ob);
    }

    public Set<Integer> getMatches(Integer boxId)
    {
        var keys = bodyKeys.get(boxId);
        var rSet = new HashSet<Integer>();
        for (BoxKey k : keys)
        {
            rSet.addAll(boxMap.get(k));
        }
        return rSet;
    }

    private BoxKey getKeyByIndex(int index_x, int index_y)
    {
        return keyMap.computeIfAbsent(index_x, SpatialMapEX::newKeyMap)
            .computeIfAbsent(index_y, (_k) -> new BoxKey(index_x, index_y));
    }
}
