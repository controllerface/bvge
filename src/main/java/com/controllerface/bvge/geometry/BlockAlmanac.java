package com.controllerface.bvge.geometry;

import org.joml.Vector2f;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BlockAlmanac
{
    private static final int ATLAS_SIZE = 32;
    private static final float UV_OFFSET = 0.03125f;
    private static final float UV_EPSILON = 0.0f;//0.0001f;

    private final List<List<Vector2f>> uv_channels;

    public BlockAlmanac()
    {
        uv_channels = Collections.unmodifiableList(generate_block_uvs());
    }

    private List<Vector2f> generate_grid_uvs(int block_location)
    {
        if (block_location < 0 || block_location >= 1024)
        {
            throw new RuntimeException("block location must be within the range 0-1023 inclusive");
        }

        int x = block_location % ATLAS_SIZE;
        int y = block_location / ATLAS_SIZE;

        float u_0 = x * UV_OFFSET + UV_EPSILON;
        float v_0 = y * UV_OFFSET + UV_EPSILON;

        float u_1 = u_0 + UV_OFFSET - UV_EPSILON;
        float v_1 = v_0 + UV_EPSILON;

        float u_2 = u_0 + UV_OFFSET - UV_EPSILON;
        float v_2 = v_0 + UV_OFFSET - UV_EPSILON;

        float u_3 = u_0 + UV_EPSILON;
        float v_3 = v_0 + UV_OFFSET - UV_EPSILON;

        var uv_channel = new ArrayList<Vector2f>();

        uv_channel.add(new Vector2f(u_0, v_0));
        uv_channel.add(new Vector2f(u_1, v_1));
        uv_channel.add(new Vector2f(u_2, v_2));
        uv_channel.add(new Vector2f(u_3, v_3));

        return uv_channel;
    }

    private List<List<Vector2f>> generate_block_uvs()
    {
        var uv_channels = new ArrayList<List<Vector2f>>();

        // todo: define with names, possibly using config data, but probably just explicit code
        uv_channels.add(generate_grid_uvs(0));
        uv_channels.add(generate_grid_uvs(1));
        uv_channels.add(generate_grid_uvs(2));
        uv_channels.add(generate_grid_uvs(3));

        return uv_channels;
    }


    public List<List<Vector2f>> uv_channels()
    {
        return uv_channels;
    }
}
