package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.ecs.components.ArmatureIndex;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.systems.physics.UniformGrid;
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

    public CameraTracking(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
    }

    @Override
    public void tick(float dt)
    {
        var focusTargets = ecs.getComponents(Component.CameraFocus);
        var focusTarget = focusTargets.entrySet().stream().findAny().orElseThrow();
        ArmatureIndex armature = Component.Armature.forEntity(ecs, focusTarget.getKey());
        if (armature == null) return;

        float[] pos = GPU.read_position(armature.index());
        float pos_x = pos[0];
        float pos_y = pos[1];

        var camera = Window.get().camera();
        var width = (float)Window.get().getWidth() * camera.get_zoom();
        var height = (float)Window.get().getHeight() * camera.get_zoom();
        var new_x = (pos_x - width / 2);
        var new_y = (pos_y - height / 2);
        var new_origin_x = (pos_x - width / camera.get_zoom()) + (uniformGrid.getWidth() / 2);
        var new_origin_y = (pos_y - height / camera.get_zoom()) + (uniformGrid.getHeight() / 2);
        camera.position.x = new_x;
        camera.position.y = new_y;
        uniformGrid.updateOrigin(new_origin_x, new_origin_y);
    }
}
