package com.controllerface.bvge.physics;

import com.controllerface.bvge.game.Sector;
import org.joml.Vector2f;

import java.util.HashSet;
import java.util.Set;

/**
 * Container class for the runtime values of a uniform grid spatial partition. The grid boundary is
 * dynamically resizable and is calculated relative to the screen dimensions.
 */
public class UniformGrid
{
    public final static int BLOCK_SIZE  = 32;
    public final static int BLOCK_COUNT = 16;
    public final static float SECTOR_SIZE = BLOCK_SIZE * BLOCK_COUNT;

    public final float perimeter_width;
    public final float perimeter_height;
    public final float outer_perimeter_width;
    public final float outer_perimeter_height;
    public final float width;
    public final float height;
    public final float inner_width;
    public final float inner_height;
    public final float outer_width;
    public final float outer_height;
    public final int x_subdivisions;
    public final int y_subdivisions;
    public final int directory_length;
    public final float x_spacing;
    public final float y_spacing;

    private float x_origin = 0;
    private float y_origin = 0;
    private float inner_x_origin = 0;
    private float inner_y_origin = 0;
    private float outer_x_origin = 0;
    private float outer_y_origin = 0;
    int key_bank_size = 0;
    int key_map_size = 0;

    private float sector_origin_x = 0;
    private float sector_origin_y = 0;
    private float sector_width = 0;
    private float sector_height = 0;

    private final Set<Sector> loaded_sectors = new HashSet<>();
    private final Vector2f world_position = new Vector2f();

    public UniformGrid(int screen_width, int screen_height)
    {
        float x = (float)screen_width * 2f;
        float y = (float)screen_height * 2f;
        perimeter_width = screen_width * .20f;
        perimeter_height = screen_height * .30f;
        outer_perimeter_width = screen_width * .10f;
        outer_perimeter_height = screen_height * .20f;
        inner_width = x;
        inner_height = y;
        width = inner_width + perimeter_width;
        height = inner_height + perimeter_height;
        outer_width = width + outer_perimeter_width;
        outer_height = height + outer_perimeter_height;
        x_subdivisions = 200;
        y_subdivisions = 100;
        directory_length = x_subdivisions * y_subdivisions;
        x_spacing = width / x_subdivisions;
        y_spacing = height / y_subdivisions;
    }

    public void update_sector_metrics(Set<Sector> loaded_sectors, float sector_origin_x, float sector_origin_y, float sector_width, float sector_height)
    {
        this.loaded_sectors.clear();
        this.loaded_sectors.addAll(loaded_sectors);
        this.sector_origin_x = sector_origin_x;
        this.sector_origin_y = sector_origin_y;
        this.sector_width    = sector_width;
        this.sector_height    = sector_height;
    }

    public void updateOrigin(float x_origin, float y_origin, float x_player, float y_player)
    {
        this.world_position.set(x_player, y_player);
        this.x_origin = x_origin;
        this.y_origin = y_origin;
        this.inner_x_origin = this.x_origin + (width - inner_width) / 2;
        this.inner_y_origin = this.y_origin + (height - inner_height) / 2;
        this.outer_x_origin = this.x_origin - outer_perimeter_width + (outer_perimeter_width / 2);
        this.outer_y_origin = this.y_origin - outer_perimeter_height + (outer_perimeter_height / 2);
    }

    public void resizeBank(int size)
    {
        key_bank_size = size;
        key_map_size = size / 2;
    }

    public int get_key_bank_size()
    {
        return key_bank_size;
    }

    public int getKey_map_size()
    {
        return key_map_size;
    }


    public float x_origin()
    {
        return x_origin;
    }

    public float y_origin()
    {
        return y_origin;
    }

    public float inner_x_origin()
    {
        return inner_x_origin;
    }

    public float inner_y_origin()
    {
        return inner_y_origin;
    }

    public float outer_x_origin()
    {
        return outer_x_origin;
    }

    public float outer_y_origin()
    {
        return outer_y_origin;
    }

    public float sector_origin_x()
    {
        return sector_origin_x;
    }

    public float sector_origin_y()
    {
        return sector_origin_y;
    }

    public float sector_width()
    {
        return sector_width;
    }

    public float sector_height()
    {
        return sector_height;
    }

    public Vector2f getWorld_position()
    {
        return world_position;
    }
}
