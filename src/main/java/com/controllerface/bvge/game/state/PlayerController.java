package com.controllerface.bvge.game.state;

import com.controllerface.bvge.core.Window;
import com.controllerface.bvge.ecs.ECS;
import com.controllerface.bvge.ecs.components.*;
import com.controllerface.bvge.editor.Editor;
import com.controllerface.bvge.events.Event;
import com.controllerface.bvge.game.Constants;
import com.controllerface.bvge.game.PlayerInput;
import com.controllerface.bvge.gpu.GPU;
import com.controllerface.bvge.gpu.GPUResource;
import com.controllerface.bvge.gpu.cl.buffers.PersistentBuffer;
import com.controllerface.bvge.gpu.cl.buffers.ResizableBuffer;
import com.controllerface.bvge.memory.sectors.SectorController;
import com.controllerface.bvge.physics.PhysicsObjects;
import com.controllerface.bvge.physics.PhysicsSimulation;
import com.controllerface.bvge.physics.UniformGrid;

import java.util.Objects;

import static com.controllerface.bvge.game.Constants.EntityFlags.*;
import static com.controllerface.bvge.game.InputBinding.*;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_float;
import static com.controllerface.bvge.gpu.cl.buffers.CL_DataTypes.cl_int;

public class PlayerController implements GPUResource
{
    private static final float BLOCK_OFFSET = UniformGrid.BLOCK_SIZE / 2.0f;
    private static final float MOTION_THRESHOLD = 10.0f;

    private final PlayerInventory player_inventory;

    private final EntityIndex entity_id;
    private final Position    position;
    private final EntityIndex mouse_cursor_id;
    private final EntityIndex block_cursor_id;
    private final FloatValue  move_force;
    private final FloatValue  jump_force;
    private final PlayerInput player;
    private final BlockCursor block_cursor;

    private final ResizableBuffer b_control_point_flags;
    private final ResizableBuffer b_control_point_indices;
    private final ResizableBuffer b_control_point_linear_mag;
    private final ResizableBuffer b_control_point_jump_mag;
    private final ResizableBuffer b_control_point_tick_budgets;

    private final StateInput input;
    private final StateOutput output;

    private final float[] entity_info_buffer = new float[SectorController.ENTITY_INFO_WIDTH];
    private final int[] empty_block_hits = new int[4];

    private final float[] block_cursor_pos;
    private final float[] entity;
    private final float[] accel;
    private final float[] current_time;
    private final float[] prev_time;
    private final float[] current_blend;
    private final short[] motion_state;
    private final int[] anim_layers;
    private final int[] prev_layers;
    private int arm_flag;
    private int current_budget;

    private BaseState current_base_state     = BaseState.IDLE;
    private MovementState current_move_state = MovementState.REST;
    private ActionState current_action_state = ActionState.NONE;

    public PlayerController(ECS ecs, PlayerInventory playerInventory)
    {
        player_inventory = playerInventory;

        b_control_point_flags        = new PersistentBuffer(GPU.compute.physics_queue, cl_int.size(), 1);
        b_control_point_indices      = new PersistentBuffer(GPU.compute.physics_queue, cl_int.size(), 1);
        b_control_point_linear_mag   = new PersistentBuffer(GPU.compute.physics_queue, cl_float.size(), 1);
        b_control_point_jump_mag     = new PersistentBuffer(GPU.compute.physics_queue, cl_float.size(), 1);
        b_control_point_tick_budgets = new PersistentBuffer(GPU.compute.physics_queue, cl_int.size(), 1);

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
        current_time     = new float[4];
        prev_time        = new float[4];
        current_blend    = new float[8];
        motion_state     = new short[2];
        anim_layers      = new int[4];
        prev_layers      = new int[4];
        arm_flag         = 0;
        current_budget   = 0;

        input  = new StateInput();
        output = new StateOutput();
    }

    private void snap_block_cursor(float x, float y)
    {
        block_cursor_pos[0] = (float) (Math.floor(x / UniformGrid.BLOCK_SIZE) * UniformGrid.BLOCK_SIZE) + BLOCK_OFFSET;
        block_cursor_pos[1] = (float) (Math.floor(y / UniformGrid.BLOCK_SIZE) * UniformGrid.BLOCK_SIZE) + BLOCK_OFFSET;
    }

    private void handle_input_states()
    {
        if (!player.pressed(MOUSE_PRIMARY))
        {
            player.unlatch_mouse();
            block_cursor.set_require_unlatch(false);
        }

        float world_x = player.get_screen_target().x * Window.get().camera().get_zoom() + Window.get().camera().position().x;
        float world_y = (Window.get().height() - player.get_screen_target().y) * Window.get().camera().get_zoom() + Window.get().camera().position().y;

        GPU.memory.update_mouse_position(mouse_cursor_id.index(), world_x, world_y);

        // todo: allow non-static/un-snapped placement using key-combo or mode switch of some kind
        snap_block_cursor(world_x, world_y);
        GPU.memory.update_block_position(block_cursor_id.index(), block_cursor_pos[0], block_cursor_pos[1]);

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
                int new_block_id = PhysicsObjects.base_block(GPU.memory.sector_container(),
                    world_x, world_y, 32, 90, 0.0f, 0.0f,
                    0, Constants.HullFlags.IS_STATIC.bits,
                    block_cursor.block(), empty_block_hits);
                GPU.memory.place_block(block_cursor_id.index(), new_block_id);
            }

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

    private int max_jump_budget = 50;
    private boolean action_layer_idle = true;
    private boolean action_layer_empty = true;

    public void update_player_state()
    {
        handle_input_states();

        GPU.memory.read_entity_info(entity_id.index(), entity_info_buffer);

        entity[0]        = entity_info_buffer[0];
        entity[1]        = entity_info_buffer[1];
        entity[2]        = entity_info_buffer[2];
        entity[3]        = entity_info_buffer[3];
        accel[0]         = entity_info_buffer[4];
        accel[1]         = entity_info_buffer[5];
        current_time[0]  = entity_info_buffer[6];
        current_time[1]  = entity_info_buffer[7];
        current_time[2]  = entity_info_buffer[8];
        current_time[3]  = entity_info_buffer[9];
        prev_time[0]     = entity_info_buffer[10];
        prev_time[1]     = entity_info_buffer[11];
        prev_time[2]     = entity_info_buffer[12];
        prev_time[3]     = entity_info_buffer[13];
        current_blend[0] = entity_info_buffer[14];
        current_blend[1] = entity_info_buffer[15];
        current_blend[2] = entity_info_buffer[16];
        current_blend[3] = entity_info_buffer[17];
        current_blend[4] = entity_info_buffer[18];
        current_blend[5] = entity_info_buffer[19];
        current_blend[6] = entity_info_buffer[20];
        current_blend[7] = entity_info_buffer[21];
        motion_state[0]  = (short)entity_info_buffer[22];
        motion_state[1]  = (short)entity_info_buffer[23];
        anim_layers[0]   = (int)entity_info_buffer[24];
        anim_layers[1]   = (int)entity_info_buffer[25];
        anim_layers[2]   = (int)entity_info_buffer[26];
        anim_layers[3]   = (int)entity_info_buffer[27];
        prev_layers[0]   = (int)entity_info_buffer[28];
        prev_layers[1]   = (int)entity_info_buffer[28];
        prev_layers[2]   = (int)entity_info_buffer[30];
        prev_layers[3]   = (int)entity_info_buffer[31];
        arm_flag         = (int)entity_info_buffer[32];

        if (Editor.ACTIVE)
        {
            Editor.queue_event("player_position", "X:" + entity[0] + " Y:" + entity[1]);
        }

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
            ? max_jump_budget // todo: this should be a player stat, it is their jump height
            : current_budget;

        input.can_jump       = can_jump;
        input.is_wet         = is_wet;
        input.can_click      = !click_disabled;
        input.current_budget = current_budget;
        input.motion_state   = motion_state;
        input.current_time   = current_time[0];
        input.anim_index     = anim_layers[0];
        input.jump_mag       = jump_force.magnitude();
        input.max_jump_budget = max_jump_budget;

        var next_base_state   = BaseState.process(input, output, current_base_state, player);
        var next_move_state   = MovementState.process(input, output, current_move_state, player);
        var next_action_state = ActionState.process(input, output, current_action_state, player);

        boolean blend_base   = current_base_state != next_base_state;
        boolean blend_move   = current_move_state != next_move_state;
        boolean blend_action = current_action_state != next_action_state;

        if (blend_base)
        {
            anim_layers[0]   = next_base_state.animation.ordinal();
            prev_layers[0]   = current_base_state.animation.ordinal();
            prev_time[0]     = current_time[0];
            current_time[0]  = 0.0f;
            current_blend[0] = AnimationState.blend_time(current_base_state.animation, next_base_state.animation);
            current_blend[1] = 0.0f;
        }

        if (blend_move)
        {
            //System.out.println("blend move: " + current_move_state + " to " + next_move_state);
            anim_layers[1]   = next_move_state.animation.ordinal();
            prev_layers[1]   = current_move_state.animation.ordinal();
            prev_time[1]     = current_time[1];
            current_time[1]  = 0.0f;
            current_blend[2] = AnimationState.blend_time(current_move_state.animation, next_move_state.animation);
            current_blend[3] = 0.0f;

            if (!blend_action && current_action_state == ActionState.NONE)
            {
                //System.out.println("opt-in 3: " + anim_layers[1] + " to: " +  prev_layers[1]);
                anim_layers[2]   = anim_layers[1];
                prev_layers[2]   = prev_layers[1];
                prev_time[2]     = prev_time[1];
                current_time[2]  = current_time[1];
                current_blend[4] = current_blend[2];
                current_blend[5] = current_blend[3];
            }
        }

        if (blend_action)
        {
            //System.out.println("blend 3: " + current_action_state + " to: " + next_action_state);
            action_layer_empty = false;
            anim_layers[2] = next_action_state.animation.ordinal();
            prev_layers[2] = current_action_state.animation.ordinal();
            prev_time[2] = current_time[2];
            current_time[2] = 0.0f;
            current_blend[4] = AnimationState.blend_time(current_action_state.animation, next_action_state.animation);
            current_blend[5] = 0.0f;
        }

        // when blending is done, unset the previous animation for the layer
        if (current_blend[1] >= current_blend[0])
        {
            // layer zero does not need a fallback
            prev_layers[0] = -1;
        }
        if (current_blend[3] >= current_blend[2])
        {
            if (prev_layers[1] != -1)
            {
                // todo: add layer 1 fallback
            }
            prev_layers[1] = -1;
        }
        if (current_blend[5] >= current_blend[4])
        {
            // layer 2 fallback
            if (!action_layer_empty && prev_layers[2] != -1 && current_action_state == ActionState.NONE && current_move_state != MovementState.REST)
            {
                System.out.println("fallback 3: " + current_action_state + " to: " + current_move_state);
                action_layer_idle = true;
                anim_layers[2] = current_move_state.animation.ordinal();
                prev_layers[2] = current_action_state.animation.ordinal();
                prev_time[2] = current_time[2];
                current_time[2] = current_time[1];
                current_blend[4] = AnimationState.blend_time(current_action_state.animation, current_move_state.animation);
                current_blend[5] = 0.0f;
            }

            if (action_layer_idle) action_layer_empty = true;

            if (!action_layer_idle) prev_layers[2] = -1;
            action_layer_idle = false;

            // todo: add layer 1 fallback
        }
        if (current_blend[7] >= current_blend[6])
        {
            if (prev_layers[3] != -1)
            {
                //System.out.println("blend end 4");
            }
            prev_layers[3] = -1;
        }


        // jumping

        if (output.jumping)
        {
            accel[1] = output.jump_amount;
            current_budget = output.next_budget;
        }
        else accel[1] = 0.0f;

        // motion state


        float vel_y = (entity[1] - entity[3]) / PhysicsSimulation.FIXED_TIME_STEP;

        motion_state[0] = (vel_y < -MOTION_THRESHOLD)
            ? (short)(motion_state[0] + 1)
            : (short)0;

        motion_state[1] = (vel_y > MOTION_THRESHOLD)
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
        if (player.pressed(MOVE_UP) && is_wet)   accel[1] = move_force.magnitude() * 2f;
        if (player.pressed(MOVE_DOWN) && is_wet) accel[1] = -move_force.magnitude();

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

        arm_flag = output.jumping
            ? arm_flag | JUMPING.bits
            : arm_flag & ~JUMPING.bits;

        //System.out.println("debug: anim_layers[] = " + Arrays.toString(anim_layers));
        //System.out.println("debug: prev_layers[] = " + Arrays.toString(prev_layers));
        //System.out.println("debug: current_blend[] = " + Arrays.toString(current_blend));


        //System.out.println("debug: base: " + current_base_state + " move: " + current_move_state + " action: " + current_action_state);
        if (anim_layers[0]==-1) anim_layers[0] =0;
        if (anim_layers[1]==-1) anim_layers[1] =0;
        if (anim_layers[2]==-1) anim_layers[2] =0;
        if (anim_layers[3]==-1) anim_layers[3] =0;

        GPU.memory.write_entity_info(entity_id.index(),
            accel,
            current_time,
            prev_time,
            current_blend,
            motion_state,
            anim_layers,
            prev_layers,
            arm_flag);

        current_base_state   = next_base_state;
        current_move_state   = next_move_state;
        current_action_state = next_action_state;
    }

    public void release()
    {
        b_control_point_flags.release();
        b_control_point_indices.release();
        b_control_point_tick_budgets.release();
        b_control_point_linear_mag.release();
        b_control_point_jump_mag.release();
    }
}
