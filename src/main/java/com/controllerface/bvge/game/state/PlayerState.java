package com.controllerface.bvge.game.state;

import com.controllerface.bvge.physics.StateOutput;

public class PlayerState
{
    private static BaseState base_state     = BaseState.IDLE;
    private static ActionState action_state = ActionState.IDLE;
    private static MovementState move_state = MovementState.IDLE;

    private static void init_output(StateOutput output)
    {
        output.jumping     = false;
        output.attack      = false;
        output.next_budget = 0;
        output.jump_amount = 0.0f;
    }

}
