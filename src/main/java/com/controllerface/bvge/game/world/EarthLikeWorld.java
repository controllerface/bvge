package com.controllerface.bvge.game.world;

import com.controllerface.bvge.game.world.sectors.Sector;
import com.controllerface.bvge.physics.PhysicsEntityBatch;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.substances.Liquid;
import com.controllerface.bvge.substances.Solid;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.util.FastNoiseLite;
import com.controllerface.bvge.util.MathEX;

import java.util.Random;

import static com.controllerface.bvge.substances.Solid.*;

public class EarthLikeWorld implements WorldType
{
    private final FastNoiseLite noise = new FastNoiseLite();
    private final FastNoiseLite noise2 = new FastNoiseLite();
    private final FastNoiseLite noise3 = new FastNoiseLite();
    private final Random random = new Random();

    private static final float block_range_floor = -0.1f;
    private static final float shard_range_floor = -0.14f;
    private static final float water_range_floor = -0.15f;

    private int map_to_block(float n, float floor, float length)
    {
        return (int) MathEX.map(n, floor, 1f, 0f, length);
    }

    public EarthLikeWorld()
    {
        noise.SetNoiseType(FastNoiseLite.NoiseType.Perlin);
        noise.SetFrequency(0.02f);
        noise.SetFractalType(FastNoiseLite.FractalType.FBm);

        noise2.SetNoiseType(FastNoiseLite.NoiseType.Cellular);
        noise2.SetFractalType(FastNoiseLite.FractalType.FBm);
        noise2.SetCellularDistanceFunction(FastNoiseLite.CellularDistanceFunction.Hybrid);

        noise3.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
        noise3.SetFractalType(FastNoiseLite.FractalType.PingPong);
    }

    private Solid rare = Solid.COAL_DEPOSIT;

    private Solid[] block_pallette = new Solid[]
        {
            GRANITE,
            GRANITE,
            LIMESTONE,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
            Solid.SCHIST,
            Solid.WHITESCHIST,
            Solid.GREENSCHIST,
            Solid.BLUESCHIST,
        };

    private Solid[] block_pallette2 = new Solid[]
        {
            Solid.MUGEARITE,
            Solid.ANDESITE,
            Solid.BASALT,
            Solid.DIORITE,
            rare,
            Solid.MUGEARITE,
            Solid.ANDESITE,
            Solid.BASALT,
            Solid.DIORITE,
            Solid.MUGEARITE,
            Solid.ANDESITE,
            Solid.BASALT,
            Solid.DIORITE,
            Solid.MUGEARITE,
            Solid.ANDESITE,
            Solid.BASALT,
            Solid.DIORITE,
        };

    private Solid[] block_pallette3 = new Solid[]
        {
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.KIMBERLITE,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.KIMBERLITE,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.COAL_DEPOSIT,
            Solid.KIMBERLITE,
        };

    public float rando_float(float baseNumber, float percentage)
    {

        float upperBound = baseNumber * percentage;
        return baseNumber + random.nextFloat() * (upperBound - baseNumber);
    }

    boolean flip = false;

    @Override
    public PhysicsEntityBatch generate_sector(Sector sector)
    {
        float x_offset = sector.x() * (int)UniformGrid.SECTOR_SIZE;
        float y_offset = sector.y() * (int)UniformGrid.SECTOR_SIZE;

        var batch = new PhysicsEntityBatch();

        for (int x = 0; x < UniformGrid.BLOCK_COUNT; x++)
        {
            for (int y = 0; y < UniformGrid.BLOCK_COUNT; y++)
            {
                float world_x = (x * UniformGrid.BLOCK_SIZE) + x_offset;
                float world_x_block = world_x + (UniformGrid.BLOCK_SIZE / 2f);
                float world_y = (y * UniformGrid.BLOCK_SIZE) + y_offset;
                float world_y_block = world_y + (UniformGrid.BLOCK_SIZE / 2f);

                if (world_y >= 0.0f) surface(batch, world_x_block, world_y_block, world_y, x, y);
                else underground(batch, world_x_block, world_y_block, world_y);
            }
        }
        return batch;
    }

    private void surface(PhysicsEntityBatch batch, float world_x_block, float world_y_block, float world_y, int x, int y)
    {
        if (world_y > 500.0) return;

        int[] hits;
        int h1 = random.nextInt(1000, 2000);
        int h2 = random.nextInt(1000, 2000);
        int h3 = random.nextInt(1000, 2000);
        int h4 = random.nextInt(1000, 2000);
        hits = new int[]{ h1, h2, h3, h4 };

        float block_x = world_x_block / UniformGrid.BLOCK_SIZE;
        float block_y = world_y / UniformGrid.BLOCK_SIZE;
        float n = noise.GetNoise(block_x, block_y);
        if (n < 0) n *= -1;
        int block = map_to_block(n, block_range_floor, (float)block_pallette.length);
        var solid = block_pallette[block];
        batch.new_block(world_x_block, world_y_block, UniformGrid.BLOCK_SIZE, 90f, 0.03f, 0.0003f, 0,
            Constants.HullFlags.IS_STATIC.bits | Constants.HullFlags.OUT_OF_BOUNDS.bits, solid, hits);
    }

    private void underground(PhysicsEntityBatch batch, float world_x_block, float world_y_block, float world_y)
    {
        float block_x = world_x_block / UniformGrid.BLOCK_SIZE;
        float block_x_2 = world_x_block / (UniformGrid.BLOCK_SIZE * 10f);
        float block_y = world_y / UniformGrid.BLOCK_SIZE;
        float block_y_2 = world_y_block / (UniformGrid.BLOCK_SIZE * .1f);

        float sz_solid = UniformGrid.BLOCK_SIZE;
        float sz_liquid = (float)UniformGrid.BLOCK_SIZE * .75f; //rando_float(UniformGrid.BLOCK_SIZE, .65f);

        float n = noise.GetNoise(block_x, block_y);
        boolean gen_block = n >= block_range_floor;
        int layer = 0;
        if (gen_block)
        {
            int block = map_to_block(n, block_range_floor, (float)block_pallette.length);
            var solid = block_pallette[block];
            if (solid != LIMESTONE && solid != GRANITE)
            {
                layer = 1;
                float n2 = Math.abs(noise2.GetNoise(block_x_2, block_y));
                block = map_to_block(n2, 0, (float)block_pallette2.length);
                solid = block_pallette2[block];
                if (solid == rare)
                {
                    layer = 2;
                    float n3 = Math.abs(noise3.GetNoise(block_x_2, block_y_2));
                    block = map_to_block(n3, 0, (float)block_pallette3.length);
                    solid = block_pallette3[block];
                }
            }

            int flags = Constants.HullFlags.IS_STATIC.bits;
            flags |= Constants.HullFlags.OUT_OF_BOUNDS.bits;

            float mn_below = noise.GetNoise(block_x, block_y - 1);
            boolean underside = mn_below < block_range_floor && mn_below > shard_range_floor;
            if (underside) batch.new_shard(true, false, world_x_block, world_y_block, sz_solid, 0, flags,.1f, 0.0f, 0.005f, ANDESITE);
            else
            {
                int[] hits;
                boolean shard = false;
                boolean flip_shard = true;

                int h1 = random.nextInt(1000, 2000);
                int h2 = random.nextInt(1000, 2000);
                int h3 = random.nextInt(100, 1000);
                int h4 = random.nextInt(100, 1000);

                hits = new int[]{ h1, h2, h3, h4 };
                //}
                if (layer == 0)
                {
                    float n_above  = noise.GetNoise(block_x, block_y + 1);
                    float n_left   = noise.GetNoise(block_x - 1, block_y);
                    float n_right  = noise.GetNoise(block_x + 1, block_y);
                    float n_uleft  = noise.GetNoise(block_x - 1, block_y + 1);
                    float n_uright = noise.GetNoise(block_x + 1, block_y + 1);
                    float n_bleft  = noise.GetNoise(block_x - 1, block_y - 1);
                    float n_bright = noise.GetNoise(block_x + 1, block_y - 1);

                    boolean air_above = n_above < block_range_floor && n_above > shard_range_floor;
                    boolean air_labove = n_uleft < block_range_floor && n_uleft > shard_range_floor;
                    boolean air_rabove = n_uright < block_range_floor && n_uright > shard_range_floor;
                    boolean air_left = n_left < block_range_floor && n_left > shard_range_floor;
                    boolean air_right = n_right < block_range_floor && n_right > shard_range_floor;

                    shard = air_above && air_left && air_labove && !air_right;
                    flip_shard = air_above && air_right && air_rabove && !air_left;

                    int block_above  = map_to_block(n_above,  block_range_floor, (float)block_pallette.length);
                    int block_below  = map_to_block(mn_below,  block_range_floor, (float)block_pallette.length);
                    int block_left   = map_to_block(n_left,   block_range_floor, (float)block_pallette.length);
                    int block_right  = map_to_block(n_right,  block_range_floor, (float)block_pallette.length);
                    int block_uleft  = map_to_block(n_uleft,  block_range_floor, (float)block_pallette.length);
                    int block_uright = map_to_block(n_uright, block_range_floor, (float)block_pallette.length);
                    int block_bleft  = map_to_block(n_bleft,  block_range_floor, (float)block_pallette.length);
                    int block_bright = map_to_block(n_bright, block_range_floor, (float)block_pallette.length);

                    h1 = 0;
                    h2 = 0;
                    h3 = 0;
                    h4 = 0;

                    h1 += block == block_below  ? 0 : 500;
                    h1 += block == block_left   ? 0 : 500;
                    h1 += block == block_bleft  ? 0 : 500;

                    h2 += block == block_below  ? 0 : 500;
                    h2 += block == block_right  ? 0 : 500;
                    h2 += block == block_bright ? 0 : 500;

                    h3 += block == block_above  ? 0 : 500;
                    h3 += block == block_right  ? 0 : 500;
                    h3 += block == block_uright ? 0 : 500;

                    h4 += block == block_above  ? 0 : 500;
                    h4 += block == block_left   ? 0 : 500;
                    h4 += block == block_uleft  ? 0 : 500;

                    hits = new int[]{ h1, h2, h3, h4 };
                }

                if (layer == 0 && (shard || flip_shard)) batch.new_shard(false, flip_shard, world_x_block, world_y_block,  sz_solid, 0,
                    Constants.HullFlags.IS_STATIC.bits | Constants.HullFlags.OUT_OF_BOUNDS.bits,
                    .1f, 0.05f, 0.005f, solid);
                else

                    batch.new_block(world_x_block, world_y_block, sz_solid, 90f, 0.03f, 0.0003f, 0,
                        Constants.HullFlags.IS_STATIC.bits | Constants.HullFlags.OUT_OF_BOUNDS.bits, solid, hits);
            }
        }
        else if (n < water_range_floor)
        {
            int hull_flags = Constants.HullFlags.IS_LIQUID.bits | Constants.HullFlags.OUT_OF_BOUNDS.bits;
            int point_flags = flip
                ? Constants.PointFlags.FLOW_LEFT.bits
                : 0;
            flip = !flip;
            batch.new_liquid(world_x_block, world_y_block,  sz_liquid, .1f, 0.0f, -0.000001f, 0, hull_flags, point_flags, Liquid.WATER);
        }
        else if (n < shard_range_floor)
        {
            batch.new_shard(true, false, world_x_block, world_y_block,  sz_solid, 0, Constants.HullFlags.OUT_OF_BOUNDS.bits,50f, 0.00f, 0.005f, Solid.BASALT);
        }
    }
}
