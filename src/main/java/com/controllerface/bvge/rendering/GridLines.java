package com.controllerface.bvge.rendering;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.Component_OLD;
import com.controllerface.bvge.window.Window;
import com.controllerface.bvge.util.Settings;
import org.joml.Vector2f;
import org.joml.Vector3f;

public class GridLines extends Component_OLD
{
    @Override
    public void update(float dt)
    {
        Camera camera = Window.getScene().camera();
        Vector2f cameraPos = camera.position;
        Vector2f projSize = camera.getProjectionSize();

        int firstX = ((int)(cameraPos.x / Settings.GRID_WIDTH) -1) * Settings.GRID_WIDTH;
        int firstY = ((int)(cameraPos.y / Settings.GRID_HEIGHT) -1) * Settings.GRID_HEIGHT;

        int numVerticals = (int)(projSize.x * camera.getZoom() / Settings.GRID_WIDTH) + 2;
        int numHorizonals = (int)(projSize.y * camera.getZoom() / Settings.GRID_HEIGHT) + 2;

        int height = (int)(projSize.y * camera.getZoom())  + Settings.GRID_HEIGHT * 2;
        int width = (int)(projSize.x * camera.getZoom()) + Settings.GRID_WIDTH * 2;

        int maxLines = Math.max(numVerticals, numHorizonals);
        Vector3f color = new Vector3f(0.2f, 0.2f, 0.2f);
        for (int i = 0; i < maxLines; i++)
        {
            int x  = firstX + (Settings.GRID_WIDTH  * i);
            int y  = firstY + (Settings.GRID_HEIGHT  * i);

            if (i < numVerticals)
            {
                DebugDraw.addLine2D(new Vector2f(x, firstY), new Vector2f(x, firstY + height), color);
            }

            if (i < numHorizonals)
            {
                DebugDraw.addLine2D(new Vector2f(firstX, y), new Vector2f(firstX + width, y), color);
            }
        }
    }
}

