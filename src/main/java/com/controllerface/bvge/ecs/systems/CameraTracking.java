package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.EntityIndex;
import com.controllerface.bvge.ecs.components.ComponentType;
import com.controllerface.bvge.ecs.components.Position;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;

import java.util.Objects;

/**
 * A simple system that ensures the game camera tracks the player. This class has an implicit assumption that
 * only one entity will have a CameraFocus component, and that this entity is the player. In the future, it
 * may be possible that this assumption is not held, however this class should still operate correctly, in that
 * _some_ element will get focused.
 */
public class CameraTracking extends GameSystem
{
    private final UniformGrid uniformGrid;
    private final float x_offset;
    private final float y_offset;

    public CameraTracking(ECS ecs, UniformGrid uniformGrid, float init_x, float init_y)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        x_offset = uniformGrid.width / 2;
        y_offset = uniformGrid.height / 2;
        update_position(init_x, init_y);
    }

    private void update_position(float x, float y)
    {
        var camera = Window.get().camera();
        var width = (float)Window.get().width() * camera.get_zoom();
        var height = (float)Window.get().height() * camera.get_zoom();
        var dx = Window.get().width() - uniformGrid.width;
        var dy = Window.get().height() - uniformGrid.height;
        var new_x = (x - width / 2);
        var new_y = (y - height / 2);
        var new_origin_x = (x - width / camera.get_zoom()) + (x_offset + dx);
        var new_origin_y = (y - height / camera.get_zoom()) + (y_offset + dy);
        camera.adjust_position(new_x, new_y);
        uniformGrid.updateOrigin(new_origin_x, new_origin_y, x, y);
    }

    @Override
    public void tick(float dt)
    {
        EntityIndex entity_id = ComponentType.EntityId.forEntity(ecs, Constants.PLAYER_ID);
        Objects.requireNonNull(entity_id);
        float[] pos = GPGPU.core_memory.read_entity_position(entity_id.index());
        float pos_x = pos[0];
        float pos_y = pos[1];
        update_position(pos_x, pos_y);
        ecs.attach_component(Constants.PLAYER_ID, ComponentType.Position, new Position(pos_x, pos_y));
    }
}
