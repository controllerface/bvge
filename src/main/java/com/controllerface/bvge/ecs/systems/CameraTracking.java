package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.cl.GPU;
import com.controllerface.bvge.data.ArmatureIndex;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.systems.physics.UniformGrid;
import com.controllerface.bvge.window.Window;

public class CameraTracking extends GameSystem
{
    private final UniformGrid uniformGrid;
    public CameraTracking(ECS ecs, UniformGrid uniformGrid)
    {
        super(ecs);
        this.uniformGrid = uniformGrid;
    }

    @Override
    public void run(float dt)
    {
        var focusTargets = ecs.getComponents(Component.CameraFocus);
        var focusTarget = focusTargets.entrySet().stream().findAny().orElseThrow();
        ArmatureIndex armature = Component.Armature.forEntity(ecs, focusTarget.getKey());
        if (armature == null) return;

        float[] pos = GPU.read_position(armature.index());

        // will be null for the first few frames while the loop is primed
        if (pos == null) return;

        float pos_x = pos[0];
        float pos_y = pos[1];

        var camera = Window.get().camera();
        var width = (float)Window.get().getWidth() * camera.getZoom();
        var height = (float)Window.get().getHeight() * camera.getZoom();
        var new_x = (pos_x - width / 2);
        var new_y = (pos_y - height / 2);
        var new_origin_x = (pos_x - width / camera.getZoom()) + (uniformGrid.getWidth() / 2);
        var new_origin_y = (pos_y - height / camera.getZoom()) + (uniformGrid.getHeight() / 2);
        camera.position.x = new_x;
        camera.position.y = new_y;
        uniformGrid.updateOrigin(new_origin_x, new_origin_y);
    }
}
