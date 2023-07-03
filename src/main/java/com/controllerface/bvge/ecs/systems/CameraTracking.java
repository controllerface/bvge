package com.controllerface.bvge.ecs.systems;

import com.controllerface.bvge.data.FTransform;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.Component;
import com.controllerface.bvge.window.Window;

public class CameraTracking extends GameSystem
{
    public CameraTracking(ECS ecs)
    {
        super(ecs);
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
        camera.position.x = new_x;
        camera.position.y = new_y;
    }
}
