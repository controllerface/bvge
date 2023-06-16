package com.controllerface.bvge.rendering;

import com.controllerface.bvge.Camera;
import com.controllerface.bvge.Component_OLD;
import com.controllerface.bvge.window.Window;
import com.controllerface.bvge.util.Constants;
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

        int firstX = ((int)(cameraPos.x / Constants.GRID_WIDTH) -1) * Constants.GRID_WIDTH;
        int firstY = ((int)(cameraPos.y / Constants.GRID_HEIGHT) -1) * Constants.GRID_HEIGHT;

        int numVerticals = (int)(projSize.x * camera.getZoom() / Constants.GRID_WIDTH) + 2;
        int numHorizonals = (int)(projSize.y * camera.getZoom() / Constants.GRID_HEIGHT) + 2;

        int height = (int)(projSize.y * camera.getZoom())  + Constants.GRID_HEIGHT * 2;
        int width = (int)(projSize.x * camera.getZoom()) + Constants.GRID_WIDTH * 2;

        int maxLines = Math.max(numVerticals, numHorizonals);
        Vector3f color = new Vector3f(0.2f, 0.2f, 0.2f);
        for (int i = 0; i < maxLines; i++)
        {
            int x  = firstX + (Constants.GRID_WIDTH  * i);
            int y  = firstY + (Constants.GRID_HEIGHT  * i);

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

