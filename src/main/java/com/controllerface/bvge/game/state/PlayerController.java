package com.controllerface.bvge.game.state;

import com.controllerface.bvge.cl.GPGPU;
import com.controllerface.bvge.cl.buffers.Destroyable;
import com.controllerface.bvge.cl.buffers.PersistentBuffer;
import com.controllerface.bvge.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.physics.*;
import com.controllerface.bvge.util.Constants;
import com.controllerface.bvge.window.Window;
import com.controllerface.bvge.window.events.Event;

import java.util.Objects;

import static com.controllerface.bvge.cl.CLData.cl_float;
import static com.controllerface.bvge.cl.CLData.cl_int;
import static com.controllerface.bvge.ecs.components.InputBinding.*;
import static com.controllerface.bvge.util.Constants.EntityFlags.*;

public class PlayerController implements Destroyable
{
    private static final float BLOCK_OFFSET = UniformGrid.BLOCK_SIZE / 2.0f;

    private final PlayerInventory player_inventory;

    private final EntityIndex entity_id;
    private final Position position;
    private final EntityIndex mouse_cursor_id;
    private final EntityIndex block_cursor_id;
    private final FloatValue move_force;
    private final FloatValue jump_force;
    private final PlayerInput player;
    private final BlockCursor block_cursor;

    private final ResizableBuffer b_control_point_flags;
    private final ResizableBuffer b_control_point_indices;
    private final ResizableBuffer b_control_point_linear_mag;
    private final ResizableBuffer b_control_point_jump_mag;
    private final ResizableBuffer b_control_point_tick_budgets;

    private final StateInput input;
    private final StateOutput output;

    private final float[] block_cursor_pos;
    private final float[] entity;
    private final float[] accel;
    private final float[] current_time;
    private final float[] current_blend;
    private final short[] motion_state;
    private final int[] anim_layers;
    private final int[] anim_previous;
    private int arm_flag;
    private int current_budget;

    public PlayerController(ECS ecs, PlayerInventory playerInventory)
    {
        player_inventory = playerInventory;

        b_control_point_flags        = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_int.size(), 1);
        b_control_point_indices      = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_int.size(), 1);
        b_control_point_linear_mag   = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_float.size(), 1);
        b_control_point_jump_mag     = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_float.size(), 1);
        b_control_point_tick_budgets = new PersistentBuffer(GPGPU.ptr_compute_queue, cl_int.size(), 1);

        this.entity_id       = ComponentType.EntityId.forEntity(ecs, Constants.PLAYER_ID);
        this.position        = ComponentType.Position.forEntity(ecs, Constants.PLAYER_ID);
        this.mouse_cursor_id = ComponentType.MouseCursorId.forEntity(ecs, Constants.PLAYER_ID);
        this.block_cursor_id = ComponentType.BlockCursorId.forEntity(ecs, Constants.PLAYER_ID);
        this.move_force      = ComponentType.MovementForce.forEntity(ecs, Constants.PLAYER_ID);
        this.jump_force      = ComponentType.JumpForce.forEntity(ecs, Constants.PLAYER_ID);
        this.player          = ComponentType.InputState.forEntity(ecs, Constants.PLAYER_ID);
        this.block_cursor    = ComponentType.BlockCursor.forEntity(ecs, Constants.PLAYER_ID);

        Objects.requireNonNull(position);
        Objects.requireNonNull(entity_id);
        Objects.requireNonNull(mouse_cursor_id);
        Objects.requireNonNull(block_cursor_id);
        Objects.requireNonNull(move_force);
        Objects.requireNonNull(jump_force);
        Objects.requireNonNull(player);
        Objects.requireNonNull(block_cursor);

        block_cursor_pos = new float[2];
        entity           = new float[4];
        accel            = new float[2];
        current_time     = new float[2];
        current_blend    = new float[2];
        motion_state     = new short[2];
        anim_layers      = new int[2];
        anim_previous    = new int[2];
        arm_flag         = 0;
        current_budget   = 0;

        input  = new StateInput();
        output = new StateOutput();
    }

    private void snap_block_cursor(float x, float y)
    {
        float _x = (float) (Math.floor(x / UniformGrid.BLOCK_SIZE) * UniformGrid.BLOCK_SIZE);
        float _y = (float) (Math.floor(y / UniformGrid.BLOCK_SIZE) * UniformGrid.BLOCK_SIZE);
        block_cursor_pos[0] = _x + BLOCK_OFFSET;
        block_cursor_pos[1] = _y + BLOCK_OFFSET;
    }

    private void handle_input_states()
    {
        if (!player.inputs().get(MOUSE_PRIMARY))
        {
            player.unlatch_mouse();
            block_cursor.set_require_unlatch(false);
        }

        var camera = Window.get().camera();
        float world_x = player.get_screen_target().x * camera.get_zoom() + camera.position().x;
        float world_y = (Window.get().height() - player.get_screen_target().y) * camera.get_zoom() + camera.position().y;
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

        // todo: allow non-static/un-snapped placement using key-combo or mode switch of some kind
        snap_block_cursor(x_pos, y_pos);
        GPGPU.core_memory.update_block_position(block_cursor_id.index(), block_cursor_pos[0], block_cursor_pos[1]);

        if (player.pressed(MOUSE_PRIMARY)
            && block_cursor.is_active()
            && !player.mouse_latched())
        {
            player.latch_mouse();

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

            // todo: may be better moving the finer details into the inventory system
            //  and just emit a new event to let it know that the count has gone below
            //  the threshold. may also want to encode the threshold somewhere so
            //  it isn't hard-coded, and could be subject to player buffs (maybe?)
            Window.get().event_bus().emit_event(Event.inventory(Event.Type.ITEM_CHANGE));
            if (resource_count < 4)
            {
                block_cursor.set_block(null);
                block_cursor.set_require_unlatch(true);
                Window.get().event_bus().emit_event(Event.select_block(null));
                Window.get().event_bus().emit_event(Event.message(Event.Type.ITEM_PLACING, "-"));
            }
        }
    }

    public void update_player_state()
    {
        handle_input_states();

        var info = GPGPU.core_memory.read_entity_info(entity_id.index());

        entity[0]        = info[0];
        entity[1]        = info[1];
        entity[2]        = info[2];
        entity[3]        = info[3];
        accel[0]         = info[4];
        accel[1]         = info[5];
        current_time[0]  = info[6];
        current_time[1]  = info[7];
        current_blend[0] = info[8];
        current_blend[1] = info[9];
        motion_state[0]  = (short)info[10];
        motion_state[1]  = (short)info[11];
        anim_layers[0]   = (int)info[12];
        anim_layers[1]   = (int)info[13];
        anim_previous[0] = (int)info[14];
        anim_previous[1] = (int)info[15];
        arm_flag         = (int)info[16];

        boolean can_jump   = (arm_flag & Constants.EntityFlags.CAN_JUMP.bits) !=0;
        boolean is_wet     = (arm_flag & Constants.EntityFlags.IS_WET.bits)   !=0;

        boolean click_disabled = false;
        if (block_cursor.is_active() || (block_cursor.requires_unlatch() && player.mouse_latched()))
        {
            click_disabled = true;
        }

        float move_mod = is_wet
            ? 0.5f
            : player.pressed(RUN)
                ? 2.0f
                : 1.0f;

        current_budget = can_jump && !player.pressed(JUMP)
            ? 10 // todo: this should be a player stat, it is their jump height
            : current_budget;

        input.can_jump       = can_jump;
        input.is_wet         = is_wet;
        input.can_click      = !click_disabled;
        input.current_budget = current_budget;
        input.motion_state   = motion_state;
        input.current_time   = current_time[0];
        input.anim_index     = anim_layers[0];
        input.jump_mag       = jump_force.magnitude();


        // todo: states from different layers need different processing. Layer 0 should
        //  always have some kind of idle animation, layer 1 any whole body animations,
        //  layer 2 upper body only animations, and layer 3 empty for now.
        var current_state = AnimationState.from_index(anim_layers[0]);
        var next_state    = AnimationState.process(input, output, current_state, player);

        // transition handling

        boolean blend = current_state != next_state;

        if (blend)
        {
            anim_previous[0] = anim_layers[0];
            current_time[1]  = current_time[0];
            current_time[0]  = 0.0f;
            current_blend[0] = AnimationState.blend_time(current_state, next_state);
            current_blend[1] = 0.0f;
        }

        anim_layers[0] = next_state.ordinal();

        // jumping

        if (output.jumping)
        {
            accel[1] = output.jump_amount;
            current_budget = output.next_budget;
        }

        // motion state

        float threshold = 10.0f;
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

        // walk/run
        if ( player.pressed(MOVE_LEFT)  && !player.pressed(MOVE_RIGHT)) accel[0] = -move_force.magnitude() * move_mod;
        if ( player.pressed(MOVE_RIGHT) && !player.pressed(MOVE_LEFT))  accel[0] = move_force.magnitude() * move_mod;
        if (!player.pressed(MOVE_RIGHT) && !player.pressed(MOVE_LEFT))  accel[0] = 0;

        // swim
        if ( player.pressed(MOVE_UP) && is_wet)   accel[1] = move_force.magnitude() * 1.5f;
        if ( player.pressed(MOVE_DOWN) && is_wet) accel[1] = -move_force.magnitude();

        arm_flag = player.pressed(MOVE_LEFT) != player.pressed(MOVE_RIGHT)
            ? player.pressed(MOVE_LEFT)
                ? arm_flag | FACE_LEFT.bits
                : arm_flag & ~FACE_LEFT.bits
            : arm_flag;

        arm_flag = output.attack
            ? arm_flag | ATTACKING.bits
            : arm_flag & ~ATTACKING.bits;

        arm_flag = player.pressed(MOUSE_SECONDARY)
            ? arm_flag | CAN_COLLECT.bits
            : arm_flag & ~CAN_COLLECT.bits;

        GPGPU.core_memory.write_entity_info(entity_id.index(),
            accel,
            current_time,
            current_blend,
            motion_state,
            anim_layers,
            anim_previous,
            arm_flag);
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
