package com.controllerface.bvge.physics;

import com.controllerface.bvge.animation.AnimationSettings;
import com.controllerface.bvge.animation.AnimationState;
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

import static com.controllerface.bvge.animation.AnimationState.*;
import static com.controllerface.bvge.cl.CLData.cl_float;
import static com.controllerface.bvge.cl.CLData.cl_int;
import static com.controllerface.bvge.util.Constants.EntityFlags.*;

public class PlayerController implements Destroyable
{
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
    private final InputState inputs;
    private final BlockCursor block_cursor;

    private final ResizableBuffer b_control_point_flags;
    private final ResizableBuffer b_control_point_indices;
    private final ResizableBuffer b_control_point_linear_mag;
    private final ResizableBuffer b_control_point_jump_mag;
    private final ResizableBuffer b_control_point_tick_budgets;

    private int current_budget = 0;

    public PlayerController(ECS ecs, PlayerInventory playerInventory)
    {
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
        this.inputs = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        this.block_cursor    = ComponentType.BlockCursor.forEntity(ecs, Constants.PLAYER_ID);

        Objects.requireNonNull(position);
        Objects.requireNonNull(entity_id);
        Objects.requireNonNull(mouse_cursor_id);
        Objects.requireNonNull(block_cursor_id);
        Objects.requireNonNull(move_force);
        Objects.requireNonNull(jump_force);
        Objects.requireNonNull(inputs);
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

    private int handle_input_states()
    {
        if (!inputs.inputs().get(InputBinding.MOUSE_PRIMARY))
        {
            inputs.unlatch_mouse();
            block_cursor.set_require_unlatch(false);
        }

        int flags = 0;

        for (var binding : InputBinding.values())
        {
            var on = inputs.inputs().get(binding);
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
                            || (block_cursor.requires_unlatch() && inputs.mouse_latched()))
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
        float world_x = inputs.get_screen_target().x * camera.get_zoom() + camera.position().x;
        float world_y = (Window.get().height() - inputs.get_screen_target().y) * camera.get_zoom() + camera.position().y;
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

        if (inputs.is_set(InputBinding.MOUSE_PRIMARY)
            && block_cursor.is_active()
            && !inputs.mouse_latched())
        {
            inputs.latch_mouse();
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
        return flags;
    }

    private void old_way()
    {
        k_handle_movement
            .set_arg(HandleMovement_k.Args.dt, PhysicsSimulation.FIXED_TIME_STEP)
            .call_task();
    }








    private static class InputData
    {
        boolean is_mv_l;
        boolean is_mv_r;
        boolean mv_jump;
        boolean mv_run;
        boolean can_jump;
        boolean is_wet;
        boolean is_click_1;
        boolean is_click_2;
        int current_budget;
        short[] motion_state;
        float current_time;
        int anim_index;
        float jump_mag;
    }

    private static class OutputData
    {
        boolean accel = false;
        boolean attack = false;
        AnimationState next_state;
        int next_budget = 0;
        float jump_amount = 0.0f;

        OutputData(AnimationState init_state)
        {
            this.next_state = init_state;
        }
    }


    OutputData init_output(AnimationState current_state)
    {
        return new OutputData(current_state);
    }

    OutputData idle_state(InputData input)
    {
        OutputData output = init_output(IDLE);
        if (inputs.is_set(InputBinding.MOVE_LEFT) || inputs.is_set(InputBinding.MOVE_RIGHT))
        {
            output.next_state = input.mv_run ? RUNNING : WALKING;
        }
        if (inputs.is_set(InputBinding.MOUSE_PRIMARY))
        {
            output.next_state = PUNCH;
        }
        if (input.can_jump && current_budget > 0 && inputs.is_set(InputBinding.JUMP))
        {
            output.next_state = RECOIL;
        }
        if (input.motion_state[0] > 100)
        {
            output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        }
        if (input.motion_state[1] > 150)
        {
            output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        }
        return output;
    }

    OutputData walking_state(InputData input)
    {
        OutputData output = init_output(WALKING);
        if (input.mv_run) output.next_state = RUNNING;
        if (!input.is_mv_l && !input.is_mv_r) output.next_state = IDLE;
        if (input.is_click_1) output.next_state = PUNCH;
        if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
        if (input.motion_state[0] > 100) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        if (input.motion_state[1] > 150) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        return output;
    }

    OutputData running_state(InputData input)
    {
        OutputData output = init_output(RUNNING);
        if (!input.mv_run) output.next_state = WALKING;
        if (!input.is_mv_l && !input.is_mv_r) output.next_state = IDLE;
        if (input.is_click_1) output.next_state = PUNCH;
        if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
        if (input.motion_state[0] > 100) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        if (input.motion_state[1] > 150) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        return output;
    }

    OutputData falling_slow_state(InputData input)
    {
        OutputData output = init_output(FALLING_SLOW);
        if (input.can_jump) output.next_state = input.motion_state[0] > 200
            ? LAND_HARD
            : LAND_SOFT;
        if (input.motion_state[0] > 200) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_FAST;
        if (input.motion_state[1] > 50) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        return output;
    }

    OutputData falling_fast_state(InputData input)
    {
        OutputData output = init_output(FALLING_FAST);
        if (input.can_jump) output.next_state = input.motion_state[0] > 200
            ? LAND_HARD
            : LAND_SOFT;
        if (input.is_wet) output.next_state = SWIM_DOWN;
        if (input.motion_state[0] < 200) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        if (input.motion_state[1] > 50) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        return output;
    }

    OutputData recoil_state(InputData input)
    {
        OutputData output = init_output(RECOIL);
        if (input.current_time > 0.15f) output.next_state = JUMPING;
        return output;
    }

    OutputData jumping_state(InputData input)
    {
        OutputData output = init_output(JUMPING);
        output.accel = true;
        output.next_budget = input.current_budget;
        int tick_slice = input.current_budget > 0 ? 1 : 0;
        output.next_budget -= tick_slice;
        output.jump_amount = tick_slice == 1
            ? input.mv_jump
            ? input.jump_mag
            : input.jump_mag / 2
            : 0;
        if (tick_slice == 0) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        return output;
    }

    OutputData in_air_state(InputData input)
    {
        OutputData output = init_output(IN_AIR);
        if (input.can_jump) output.next_state = input.motion_state[0] > 200
            ? LAND_HARD
            : LAND_SOFT;
        if (input.motion_state[0] > 50) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        return output;
    }

    OutputData swim_up_state(InputData input)
    {
        OutputData output = init_output(SWIM_UP);
        if (input.can_jump) output.next_state = input.motion_state[0] > 200
            ? LAND_HARD
            : LAND_SOFT;
        if (input.motion_state[0] > 50) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        return output;
    }

    OutputData swim_down_state(InputData input)
    {
        OutputData output = init_output(SWIM_DOWN);
        if (input.can_jump) output.next_state = input.motion_state[0] > 200
            ? LAND_HARD
            : LAND_SOFT;
        if (input.motion_state[0] > 200) output.next_state = input.is_wet ? SWIM_DOWN : FALLING_SLOW;
        if (input.motion_state[1] > 50) output.next_state = input.is_wet ? SWIM_UP : IN_AIR;
        return output;
    }

    OutputData land_soft_state(InputData input)
    {
        OutputData output = init_output(LAND_SOFT);
        if (input.current_time > 0.08f) output.next_state = IDLE;
        return output;
    }

    OutputData land_hard_state(InputData input)
    {
        OutputData output = init_output(LAND_HARD);
        if (input.current_time > 0.22f) output.next_state = IDLE;
        return output;
    }

    OutputData punch_state(InputData input)
    {
        OutputData output = init_output(PUNCH);
        if (!input.is_click_1) output.next_state = (input.is_mv_l || input.is_mv_r) ? WALKING : IDLE;
        if (input.can_jump && input.current_budget > 0 && input.mv_jump) output.next_state = RECOIL;
        if (output.next_state == PUNCH) output.attack = true;
        return output;
    }




    private void new_way(int flags)
    {
        var info = GPGPU.core_memory.read_entity_info(entity_id.index());

        float current_linear_mag = move_force.magnitude();
        float current_jump_mag   = jump_force.magnitude();

        float[] entity        = new float[]{info[0], info[1], info[2], info[3]};
        float[] accel         = new float[]{info[4], info[5]};
        float[] current_time  = new float[]{info[6], info[7]};
        float[] current_blend = new float[]{info[8], info[9]};
        short[] motion_state  = new short[]{(short)info[10], (short)info[11]};
        int[] anim_index      = new int[]{(int)info[12], (int)info[13]};
        int arm_flag          = (int)info[14];

        boolean is_mv_l    = (flags & Constants.ControlFlags.LEFT.bits)       !=0;
        boolean is_mv_r    = (flags & Constants.ControlFlags.RIGHT.bits)      !=0;
        boolean is_mv_u    = (flags & Constants.ControlFlags.UP.bits)         !=0;
        boolean is_mv_d    = (flags & Constants.ControlFlags.DOWN.bits)       !=0;
        boolean is_click_1 = (flags & Constants.ControlFlags.MOUSE1.bits)     !=0;
        boolean is_click_2 = (flags & Constants.ControlFlags.MOUSE2.bits)     !=0;
        boolean mv_jump    = (flags & Constants.ControlFlags.JUMP.bits)       !=0;
        boolean mv_run     = (flags & Constants.ControlFlags.RUN.bits)        !=0;
        boolean can_jump   = (arm_flag & Constants.EntityFlags.CAN_JUMP.bits) !=0;
        boolean is_wet     = (arm_flag & Constants.EntityFlags.IS_WET.bits)   !=0;

        float move_mod = is_wet
            ? 0.5f
            : mv_run
                ? 2.0f
                : 1.0f;

        current_budget = can_jump && !mv_jump
            ? 10 // todo: this should be a player stat, it is their jump height
            : current_budget;

        InputData input = new InputData();
        input.is_mv_l        = is_mv_l;
        input.is_mv_r        = is_mv_r;
        input.mv_jump        = mv_jump;
        input.mv_run         = mv_run;
        input.can_jump       = can_jump;
        input.is_wet         = is_wet;
        input.is_click_1     = is_click_1;
        input.is_click_2     = is_click_2;
        input.current_budget = current_budget;
        input.motion_state   = motion_state;
        input.current_time   = current_time[0];
        input.anim_index     = anim_index[0];
        input.jump_mag       = current_jump_mag;

        var current_state = AnimationState.from_index(anim_index[0]);
        OutputData state_result = switch (current_state)
        {
            case IDLE         -> idle_state(input);
            case WALKING      -> walking_state(input);
            case RUNNING      -> running_state(input);
            case FALLING_SLOW -> falling_slow_state(input);
            case FALLING_FAST -> falling_fast_state(input);
            case RECOIL       -> recoil_state(input);
            case JUMPING      -> jumping_state(input);
            case IN_AIR       -> in_air_state(input);
            case SWIM_UP      -> swim_up_state(input);
            case SWIM_DOWN    -> swim_down_state(input);
            case LAND_SOFT    -> land_soft_state(input);
            case LAND_HARD    -> land_hard_state(input);
            case PUNCH        -> punch_state(input);
            case UNKNOWN      -> new OutputData(UNKNOWN);
        };

        // transition handling

        boolean blend = current_state != state_result.next_state;

        current_time[1] = blend
            ? current_time[0]
            : current_time[1];

        anim_index[1] = blend
            ? anim_index[0]
            : anim_index[1];

        current_blend = blend
            ? new float[]{ AnimationSettings.blend_time(current_state, state_result.next_state) , 0.0f }
            : current_blend;

        current_time[0] = blend
            ? 0.0f
            : current_time[0];

        anim_index[0] = state_result.next_state.ordinal();

        // jumping

        accel[1] = state_result.accel
            ? state_result.jump_amount
            : accel[1];

        current_budget = state_result.accel
            ? state_result.next_budget
            : current_budget;

        // motion state

        float threshold = 10.0f;
        //float vel_x = (entity[0] - entity[2]) / PhysicsSimulation.FIXED_TIME_STEP;
        float vel_y = (entity[1] - entity[3]) / PhysicsSimulation.FIXED_TIME_STEP;


        motion_state[0] = (vel_y < -threshold)
            ? (short)(motion_state[0] + 1)
            : (short)0;

        motion_state[1] = (vel_y > threshold)
            ? (short)(motion_state[1] + 1)
            : (short)0;

        motion_state[0] = motion_state[0] > 1000
            ? 1000
            : motion_state[0];

        motion_state[1] = motion_state[1] > 1000
            ? 1000
            : motion_state[1];

        // acceleration

        accel[0] = is_mv_l && !is_mv_r
            ? -current_linear_mag * move_mod
            : accel[0];

        accel[0] = is_mv_r && !is_mv_l
            ? current_linear_mag * move_mod
            : accel[0];

        accel[0] = !is_mv_r && !is_mv_l
            ? 0.0f
            : accel[0];

        accel[1] = is_mv_u && is_wet
            ? current_linear_mag * 1.5f
            : accel[1];

        accel[1] = is_mv_d && is_wet
            ? -current_linear_mag
            : accel[1];

        // set flags

        arm_flag = is_mv_l != is_mv_r
            ? is_mv_l
                ? arm_flag | FACE_LEFT.bits
                : arm_flag & ~FACE_LEFT.bits
            : arm_flag;

        arm_flag = state_result.attack
            ? arm_flag | ATTACKING.bits
            : arm_flag & ~ATTACKING.bits;

        arm_flag = is_click_2
            ? arm_flag | CAN_COLLECT.bits
            : arm_flag & ~CAN_COLLECT.bits;

        GPGPU.core_memory.write_entity_info(entity_id.index(),
            accel,
            current_time,
            current_blend,
            motion_state,
            anim_index,
            arm_flag);
    }

    public void update_player_state()
    {
        var f = handle_input_states();
        new_way(f);
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
