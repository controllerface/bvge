package com.controllerface.bvge.physics;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.buffers.CoreBufferType;
import com.controllerface.bvge.cl.buffers.Destroyable;
import com.controllerface.bvge.cl.buffers.PersistentBuffer;
import com.controllerface.bvge.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.cl.kernels.GPUKernel;
import com.controllerface.bvge.cl.kernels.HandleMovement_k;
import com.controllerface.bvge.cl.kernels.Kernel;
import com.controllerface.bvge.cl.kernels.SetControlPoints_k;
import com.controllerface.bvge.cl.programs.ControlEntities;
import com.controllerface.bvge.cl.programs.GPUProgram;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.game.state.PlayerInventory;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import com.controllerface.bvge.window.events.Event;

import java.util.Objects;

import static com.controllerface.bvge.cl.CLData.cl_float;
import static com.controllerface.bvge.cl.CLData.cl_int;

public class PlayerController implements Destroyable
{
    private final ECS ecs;
    private final PlayerInventory player_inventory;

    private final GPUProgram p_control_entities = new ControlEntities();
    private final GPUKernel k_set_control_points;
    private final GPUKernel k_handle_movement;

    private final EntityIndex entity_id;
    private final Position position;
    private final EntityIndex mouse_cursor_id;
    private final EntityIndex block_cursor_id;
    private final FloatValue move_force;
    private final FloatValue jump_force;
    private final InputState input_state;
    private final BlockCursor block_cursor;

    private final ResizableBuffer b_control_point_flags;
    private final ResizableBuffer b_control_point_indices;
    private final ResizableBuffer b_control_point_linear_mag;
    private final ResizableBuffer b_control_point_jump_mag;
    private final ResizableBuffer b_control_point_tick_budgets;

    public PlayerController(ECS ecs, PlayerInventory playerInventory)
    {
        this.ecs = ecs;
        player_inventory = playerInventory;

        p_control_entities.init();

        b_control_point_flags        = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_int.size(), 1);
        b_control_point_indices      = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_int.size(), 1);
        b_control_point_linear_mag   = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_float.size(), 1);
        b_control_point_jump_mag     = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_float.size(), 1);
        b_control_point_tick_budgets = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_int.size(), 1);

        long k_ptr_set_control_points = p_control_entities.kernel_ptr(Kernel.set_control_points);
        k_set_control_points = new SetControlPoints_k(GPGPU.ptr_compute_queue, k_ptr_set_control_points)
            .buf_arg(SetControlPoints_k.Args.flags,      b_control_point_flags)
            .buf_arg(SetControlPoints_k.Args.indices,    b_control_point_indices)
            .buf_arg(SetControlPoints_k.Args.linear_mag, b_control_point_linear_mag)
            .buf_arg(SetControlPoints_k.Args.jump_mag,   b_control_point_jump_mag);

        long k_ptr_handle_movements = p_control_entities.kernel_ptr(Kernel.handle_movement);
        k_handle_movement = new HandleMovement_k(GPGPU.ptr_compute_queue, k_ptr_handle_movements)
            .buf_arg(HandleMovement_k.Args.entities,                 GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY))
            .buf_arg(HandleMovement_k.Args.entity_accel,             GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ACCEL))
            .buf_arg(HandleMovement_k.Args.entity_motion_states,     GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_MOTION_STATE))
            .buf_arg(HandleMovement_k.Args.entity_flags,             GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_FLAG))
            .buf_arg(HandleMovement_k.Args.entity_animation_indices, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_INDEX))
            .buf_arg(HandleMovement_k.Args.entity_animation_elapsed, GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_ELAPSED))
            .buf_arg(HandleMovement_k.Args.entity_animation_blend,   GPGPU.core_memory.get_buffer(CoreBufferType.ENTITY_ANIM_BLEND))
            .buf_arg(HandleMovement_k.Args.flags,                    b_control_point_flags)
            .buf_arg(HandleMovement_k.Args.indices,                  b_control_point_indices)
            .buf_arg(HandleMovement_k.Args.tick_budgets,             b_control_point_tick_budgets)
            .buf_arg(HandleMovement_k.Args.linear_mag,               b_control_point_linear_mag)
            .buf_arg(HandleMovement_k.Args.jump_mag,                 b_control_point_jump_mag);

        this.entity_id       = ComponentType.EntityId.forEntity(ecs, Constants.PLAYER_ID);
        this.position        = ComponentType.Position.forEntity(ecs, Constants.PLAYER_ID);
        this.mouse_cursor_id = ComponentType.MouseCursorId.forEntity(ecs, Constants.PLAYER_ID);
        this.block_cursor_id = ComponentType.BlockCursorId.forEntity(ecs, Constants.PLAYER_ID);
        this.move_force      = ComponentType.MovementForce.forEntity(ecs, Constants.PLAYER_ID);
        this.jump_force      = ComponentType.JumpForce.forEntity(ecs, Constants.PLAYER_ID);
        this.input_state     = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        this.block_cursor    = ComponentType.BlockCursor.forEntity(ecs, Constants.PLAYER_ID);

        Objects.requireNonNull(position);
        Objects.requireNonNull(entity_id);
        Objects.requireNonNull(mouse_cursor_id);
        Objects.requireNonNull(block_cursor_id);
        Objects.requireNonNull(move_force);
        Objects.requireNonNull(jump_force);
        Objects.requireNonNull(input_state);
        Objects.requireNonNull(block_cursor);
    }

    private float[] snap(float x, float y)
    {
        float offset = UniformGrid.BLOCK_SIZE / 2.0f;
        float _x = (float) (Math.floor(x / UniformGrid.BLOCK_SIZE) * UniformGrid.BLOCK_SIZE);
        float _y = (float) (Math.floor(y / UniformGrid.BLOCK_SIZE) * UniformGrid.BLOCK_SIZE);
        _x += offset;
        _y += offset;
        return new float[]{_x, _y};
    }

    public void update_player_state()
    {
        if (!input_state.inputs().get(InputBinding.MOUSE_PRIMARY))
        {
            input_state.unlatch_mouse();
            block_cursor.set_require_unlatch(false);
        }

        int flags = 0;

        for (var binding : InputBinding.values())
        {
            var on = input_state.inputs().get(binding);
            if (on)
            {
                int flag = switch (binding)
                {
                    case MOVE_UP -> Constants.ControlFlags.UP.bits;
                    case MOVE_DOWN -> Constants.ControlFlags.DOWN.bits;
                    case MOVE_LEFT -> Constants.ControlFlags.LEFT.bits;
                    case MOVE_RIGHT -> Constants.ControlFlags.RIGHT.bits;
                    case JUMP -> Constants.ControlFlags.JUMP.bits;
                    case RUN -> Constants.ControlFlags.RUN.bits;
                    case MOUSE_PRIMARY ->
                    {
                        if (block_cursor.is_active()
                            || (block_cursor.requires_unlatch() && input_state.mouse_latched()))
                        {
                            yield 0;
                        }
                        else
                        {
                            yield Constants.ControlFlags.MOUSE1.bits;
                        }
                    }
                    case MOUSE_SECONDARY -> Constants.ControlFlags.MOUSE2.bits;
                    case MOUSE_MIDDLE,
                         MOUSE_BACK,
                         MOUSE_FORWARD -> 0;
                };
                flags |= flag;
            }
        }

        // todo: probably don't need to use buffers at all for this kernel, should just pass
        //  args directly as this is only used for the player entity.
        k_set_control_points
            .set_arg(SetControlPoints_k.Args.target, 0)
            .set_arg(SetControlPoints_k.Args.new_flags, flags)
            .set_arg(SetControlPoints_k.Args.new_index, entity_id.index())
            .set_arg(SetControlPoints_k.Args.new_jump_mag, jump_force.magnitude())
            .set_arg(SetControlPoints_k.Args.new_move_mag, move_force.magnitude())
            .call_task();

        var camera = Window.get().camera();
        float world_x = input_state.get_screen_target().x * camera.get_zoom() + camera.position().x;
        float world_y = (Window.get().height() - input_state.get_screen_target().y) * camera.get_zoom() + camera.position().y;
        GPGPU.core_memory.update_mouse_position(mouse_cursor_id.index(), world_x, world_y);

        float x_pos;
        float y_pos;
        if (block_cursor.is_active())
        {
            x_pos = world_x;
            y_pos = world_y;
        }
        else
        {
            x_pos = position.x;
            y_pos = position.y;
        }

        float[] sn = snap(x_pos, y_pos);
        GPGPU.core_memory.update_block_position(block_cursor_id.index(), sn[0], sn[1]);

        if (input_state.inputs().get(InputBinding.MOUSE_PRIMARY)
            && block_cursor.is_active()
            && !input_state.mouse_latched())
        {
            input_state.latch_mouse();
            // todo: allow non-static placement using key-combo or mode switch of some kind
            int resource_count = player_inventory.solid_counts().get(block_cursor.block());
            if (resource_count >= 4)
            {
                resource_count -= 4;
                player_inventory.solid_counts().put(block_cursor.block(), resource_count);
                int new_block_id = PhysicsObjects.base_block(GPGPU.core_memory.sector_container(),
                    world_x, world_y, 32, 90, 0.0f, 0.0f,
                    0, Constants.HullFlags.IS_STATIC.bits,
                    block_cursor.block(), new int[4]);
                GPGPU.core_memory.place_block(block_cursor_id.index(), new_block_id);
            }
            Window.get().event_bus().emit_event(Event.inventory(Event.Type.ITEM_CHANGE));
            if (resource_count < 4)
            {
                // todo: may be better moving the finer details into the inventory system
                //  and just emit a new event to let it know that the count has gone below
                //  the threshold. may also want to encode the threshold somewhere so
                //  it isn't hard-coded, and could be subject to player buffs (maybe?)
                block_cursor.set_block(null);
                block_cursor.set_require_unlatch(true);
                Window.get().event_bus().emit_event(Event.select_block(null));
                Window.get().event_bus().emit_event(Event.message(Event.Type.ITEM_PLACING, "-"));
            }
        }

        k_handle_movement
            .set_arg(HandleMovement_k.Args.dt, PhysicsSimulation.FIXED_TIME_STEP)
            .call_task();
    }

    public void destroy()
    {
        b_control_point_flags.release();
        b_control_point_indices.release();
        b_control_point_tick_budgets.release();
        b_control_point_linear_mag.release();
        b_control_point_jump_mag.release();
    }
}
