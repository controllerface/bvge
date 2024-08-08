package com.controllerface.bvge.game.world;

import com.controllerface.bvge.memory.sectors.Sector;
import com.controllerface.bvge.physics.PhysicsEntityBatch;

/**
 * An interface for 2D game world implementations. The term "world" should be interpreted loosely,
 * as implementations are not expected to be bound by any rules in terms of what the "world" may
 * contain. So for example, one world could be as small as a single room or house, etc. and
 * others could represent an entire planet, or even some near-infinite sized area.
 */
public interface WorldType
{
    /**
     * From a given Sector, returns a batch of entities that exist within it. The intent is that
     * as a player enters a certain part of the world, the portion of the world that is new loaded
     * (i.e. a sector) is provided to the world implementation, and the resulting batch of entities
     * is then loaded into the active game state.
     * The dimensions of a Sector are not explicitly defined here but are the responsibility of the
     * implementing class, however, the {@linkplain com.controllerface.bvge.physics.UniformGrid#SECTOR_SIZE}
     * value is mosty likely to be used.
     * todo: the sector data might need to be pulled into a util or constants class for easier visibility
     *
     * @param sector the sector for which loading is being requested
     * @return a batch of entities that are present in the given sector
     */
    PhysicsEntityBatch generate_sector(Sector sector);
}
