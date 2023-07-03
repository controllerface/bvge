# List of things to do (in no particular order)

1. camera tracking for player
   1. when near edge of viewport, scroll to ensure player is on screen 
   2. distance from edge should be sufficient to see upcoming objects
2. static geometry
   1. physics objects should be able to be toggled static on/off
   2. add check to ensure static objects only check non-static for collision
3. circle objects
   1. will need circle/circle and circle/poly collision reaction functions
4. non-reacting flag
   1. objects should be able to collide but not produce reactions
   2. for example, bullets or other projectiles than can impact without reaction
5. spatial partition improvements
   1. objects outside tracked zone should not move or collide
   2. zone should follow player
   3. will need offset calculations to handle negative values
   4. need at least one outer tier just outside viewing zone for "settling"
   5. tiers should help "ease" objects out as player moves away
6. gravity
   1. objects should fall when gravity is enabled
   2. consider "zones" which could be used to implement planet-side mechanics
   3. radial zone for small space bound objects (asteroids, etc.) amy be cool
7. collision layers
   1. physics objects should have a layer parameter to restrict collisions
   2. layer 0 should be an "always collide" layer so things don't clip out
   3. layer -1 should be a "never collide" layer for "ethereal" objects
8. mouse cursor
   1. hide normal one and draw a custom one
9. rotation improvements
   1. need a way to restrict rotation via a flag, so players/NPCs can be upright
   2. should be a toggle, so they CAN rotate, for rag-doll moments
   3. would be useful to be able to measure current rotation angle
   4. may need some kind of "compass edge" to enable this