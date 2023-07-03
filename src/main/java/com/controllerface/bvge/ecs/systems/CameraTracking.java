package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.data.FTransform;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.ecs.systems.physics.SpatialPartition;
import com.controllerface.bvge.window.Window;

public class CameraTracking extends GameSystem
{
    private final SpatialPartition spatialPartition;
    public CameraTracking(ECS ecs, SpatialPartition spatialPartition)
    {
        super(ecs);
        this.spatialPartition = spatialPartition;
    }

    @Override
    public void run(float dt)
    {
        var focusTargets = ecs.getComponents(Component.CameraFocus);
        var focusTarget = focusTargets.entrySet().stream().findAny().orElseThrow();
        var t = ecs.getComponentFor(focusTarget.getKey(), Component.Transform);
        FTransform transform = Component.Transform.coerce(t);
        if (transform == null) return;
        var camera = Window.get().camera();
        var width = (float)Window.get().getWidth() * camera.getZoom();
        var height = (float)Window.get().getHeight() * camera.getZoom();
        var new_x = (transform.pos_x() - width / 2);
        var new_y = (transform.pos_y() - height / 2);
        var w = (transform.pos_x() - width / camera.getZoom()) + (spatialPartition.getWidth() / 2);
        var h = (transform.pos_y() - height / camera.getZoom()) + (spatialPartition.getHeight() / 2);
        camera.position.x = new_x;
        camera.position.y = new_y;
        spatialPartition.updateOrigin(w, h );
    }
}
