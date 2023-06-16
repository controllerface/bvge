package com.controllerface.bvge;

import imgui.ImGui;

import java.util.ArrayList;
import java.util.List;

public class GameObject
{
    private static int ID_COUNTER = 0;
    private int uid = -1;
    private String name;
    private List<Component_OLD> components;
    public transient Transform transform;
    private boolean doSerialization = true;

    public GameObject(String name)
    {
        this.name = name;
        this.components = new ArrayList<>();
        this.uid = ID_COUNTER ++;
    }

    public List<Component_OLD> getAllComponents()
    {
        return components;
    }

    public <T extends Component_OLD> T getComponent(Class<T> componentClass)
    {
        for (Component_OLD c : components)
        {
            if (componentClass.isAssignableFrom(c.getClass()))
            {
                try
                {
                    return componentClass.cast(c);
                }
                catch (ClassCastException e)
                {
                    e.printStackTrace();
                    assert false : "Can't cast component";
                }
            }
        }
        return null;
    }

    public <T extends Component_OLD> void removeComponent(Class<T> componentClass)
    {
        for (int i=0; i < components.size(); i++)
        {
            Component_OLD c = components.get(i);
            if (componentClass.isAssignableFrom(c.getClass()))
            {
                components.remove(i);
                return;
            }
        }
    }

    public void addComponent(Component_OLD c)
    {
        c.generateId();
        this.components.add(c);
        c.gameObject = this;
    }

    public void update(float dt)
    {
        for (int i=0;i<components.size();i++)
        {
            components.get(i).update(dt);
        }
    }

    public void start()
    {
        for (int i=0;i<components.size();i++)
        {
            components.get(i).start();
        }
    }

    public void imgui()
    {
        for (Component_OLD c : components)
        {
            if (ImGui.collapsingHeader(c.getClass().getSimpleName()))
            {
                c.imgui();
            }
        }

    }

    public static void init(int maxId)
    {
        ID_COUNTER = maxId;
    }

    public int getUid()
    {
        return this.uid;
    }

    public void setNoSerialize()
    {
        this.doSerialization = false;
    }

    public boolean doSerialization()
    {
        return this.doSerialization;
    }

}
