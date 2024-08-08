package com.controllerface.bvge.game.state;

public class PlayerState
{
    private static BaseState base_state     = BaseState.IDLE;
    private static ActionState action_state = ActionState.NONE;
    private static MovementState move_state = MovementState.REST;

    private static void init_output(StateOutput output)
    {
        output.jumping     = false;
        output.attack      = false;
        output.next_budget = 0;
        output.jump_amount = 0.0f;
    }

}
