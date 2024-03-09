package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.components.ArmatureIndex;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.physics.UniformGrid;
import com.controllerface.bvge.window.Window;

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

    public CameraTracking(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
        x_offset = uniformGrid.width / 2;
        y_offset = uniformGrid.height / 2;
    }

    @Override
    public void tick(float dt)
    {
        var focusTargets = ecs.getComponents(Component.CameraFocus);
        var focusTarget = focusTargets.entrySet().stream().findAny().orElseThrow();
        ArmatureIndex armature = Component.Armature.forEntity(ecs, focusTarget.getKey());
        if (armature == null) return;

        float[] pos = GPU.core_memory.read_position(armature.index());
        float pos_x = pos[0];
        float pos_y = pos[1];
        var camera = Window.get().camera();
        var width = (float)Window.get().getWidth() * camera.get_zoom();
        var height = (float)Window.get().getHeight() * camera.get_zoom();
        var dx = Window.get().getWidth() - uniformGrid.width;
        var dy = Window.get().getHeight() - uniformGrid.height;
        var new_x = (pos_x - width / 2);
        var new_y = (pos_y - height / 2);
        var new_origin_x = (pos_x - width / camera.get_zoom()) + (x_offset + dx);
        var new_origin_y = (pos_y - height / camera.get_zoom()) + (y_offset + dy);
        camera.position.x = new_x;
        camera.position.y = new_y;
        uniformGrid.updateOrigin(new_origin_x, new_origin_y);
    }
}
