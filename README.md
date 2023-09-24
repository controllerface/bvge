
BVGE Prototype Design Notes
=
_bastard video game engine_

Tech Stack
-
- **JDK / [Java](https://docs.oracle.com/en/java/javase/21/docs/api/index.html)** 
  - CPU Game Loop
- **Open GL / [GLSL](https://www.khronos.org/opengl/wiki/Core_Language_(GLSL))**
  - GPU Rendering 
- **Open CL / [CL C](https://man.opencl.org/)**
  - GPU Compute

Goal Summary
-
Eventually, I intend to meander this code toward something game-like, but for the immediate future, the main goal is to create a basic prototype game engine that includes a physics simulation, rendering system, and some basic input handling for controlling a player character. 

I also want to ensure that the prototype, while not fully-featured in any sense of the word, does have one _complete_ component that can be used in a later alpha phase, before I would consider the prototype "done". This component is what I generally refer to as GPU-CRUD or in other words, the full set of create, read, update, and delete operations targeting objects stored in memory on the GPU.

At a high level, the key thing that is needed is the ability to spawn, as well as de-spawn, some arbitrary "entities". A player is an entity, an enemy or NPC is also, and so is a rock or tree or really, anything that exists in the game world. 

This design is meant to support an ECS layout, which is a paradigm that is used in game development that serves a somewhat analogous purpose as MVC (model, view, controller) does in front end frameworks. 

General Layout
-
Below are some basic points about how the engine is currently laid out. As this is a prototype, details are subject to change, but I expect for the foreseeable future the following notes will generally be accurate.

### Core Classes

There are a few core classes that comprise most of the important functionality in the engine. This is not an exhaustive list, but represents some of the most vital functionality:

- [ECS](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/ecs/ECS.java): Entity Component System
  - This is a container class that provides a mechanism for creating tracked entities, giving those entities values by way of attaching components to them, and then defining systems that operate on these entities and their components.
  - Every tracked object in the engine is an entity, and entities are stored in memory as a standard [String](https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/String.html), so that unique ID is all that is needed to reference that entity within the methods provided by the ECS class. As such, there is no specific "entity" class.
  - [GameComponent](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/ecs/components/GameComponent.java)s are attached to entities and contain some value. The value is open-ended, so may be any kind of class, as long as it implements the required interface. Only a few components are defined, but the way the design works, adding new components should not affect performance, as all looks ups use hash maps and categorized queries. Generally speaking, looping over components or entities is discouraged and could be forbidden at some point.
  - [GameSystem](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/ecs/systems/GameSystem.java)s are effectively mini programs that are run in a defined order every frame. Systems implement an interface and then are registered to ensure they are run during the game loop. The intention of this design is to implement core mechanics of the game engine as systems.


- [GPU](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/cl/GPU.java): Generalized GPU Computing
  - Primarily used for physics calculations, this class represents the interface between the CPU and GPU and defines all the kernel entry points that can be called.
  - Also includes some interop functions to allow Open GL and Open CL to share data without requiring round-trips to the CPU.
  - This class is used in tandem with the Main.Memory utils to ensure that the CPU and GPU are kept in sync for data counts and offsets


- [PhysicsSimulation](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/ecs/systems/physics/PhysicsSimulation.java): Core System
  - A core system that implements the physics simulation that the game runs within. This class leans heavily on the GPU utility to call kernel functions on the GPU, which are what drives the simulation.
  - There is a concept of a "tick" which is disconnected from a frame. The simulation runs at `60 fps`, but each frame is broken into sub-steps (currently `4`) and the physics simulation is "ticked" once for each sub-step. If sub-steps were disabled, the simulation tick-rate and frame-rate would be the same, but since we have `4` sub-steps the tick rate is effectively `60` frames * `4` steps, i.e. `240 fps`.
  - Physics objects are represented as collections of 5 constituent objects:
     - **armature**: the top-level object, every entity has exactly 1 armature
       - the armature is effectively a "copy" or instance of a model that ahs been loaded.
       - many copies of the same model are individually spawned with a unique Armature, but the same reference model.
     - **hull**: an armature contains 1 or more hulls, which represent the space the object takes up
       - these are the primary objects that are used in collision checks
     - **bone**: bones are 4x4 transformation matrices that are used to animate hulls
       - the loaded model will have reference frames that are used to modify the bones, making the armature animate.
       - *animation is not fully implemented yet*
     - **point**: a hull has 1 or more points, which define the extents of the hull in physical space
       - generally, only circles will have a single point, and all other objects will have 3 or more
       - points are associated with bones providing an easy way to perform animations directly
     - **edge**: a hull has 0 or more edges, which define the edges of the hull that make it a rigid body
       - circles are the only objects that don't have an edge, their boundary is defined by their radius
       - edges are recorded as constraints which must be maintained
       - there is a call made during the physics tick that enforces these constraints


- [PhysicsObjects](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/data/PhysicsObjects.java): Spawning Utils
  - A utility class that provides functions for spawning objects in the game world
  - The goal is to have a simple generic way to create physics objects
  - creation functions return a single ID representing the tracked armature
  - the armature ID is attached to an entity as a component


- [Main.Memory](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/Main.java): CPU/GPU Boundary
  - This subclass of Main provides a centralized point where CPU code (Java) can interact with memory buffers created on the GPU
  - When a new object is spawned, this subclass delegates to calls to the GPU, which actually puts the object into memory
  - the current design requires objects are be created all at once, essentially atomically
  - this ensures that buffers remain "aligned", vital when entities must be deleted
  - THIS MECHANISM IS NOT THREAD SAFE! ONLY ONE THREAD CAN CREATE OBJECTS AT A TIME!


- [Models](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/geometry/Models.java): FBX model Loader
  - Provides utils for loading models
  - Currently loads all models inside `init()` for use by other components
  - All known models have a unique ID assigned
  - When new armatures are spawned, they generally use a loaded model as a reference and use the model ID as a tag 


- [TestGame](https://github.com/controllerface/bvge/blob/main/src/main/java/com/controllerface/bvge/game/TestGame.java): As it says...
  - This is the current "test bed" that loads all the systems, spawns the player and other objects, and just generally gets things going
  - `load()` is the main place where I generally spawn different things to test out the effects of changes

### GPU Kernels

Aside from the CPU side Java code, a number of Open CL kernels have been written, mostly to implement the physics simulation. In Open CL, code is compiled into programs, which may expose 1 or more `kernels`, which are the functions that are callable from the CPU. 

All the kernels are required for the simulation to run, but some are more central to the systems operation, where others are generally more in a "supporting" role providing some logic to set up buffers or other secondary tasks.

here is a short list of a few of the more vital kernels:

- [gpu_crud.cl](https://github.com/controllerface/bvge/blob/main/src/main/resources/cl/kernels/gpu_crud.cl): Create, Read, Update, Delete; On GPU
    - Critical functionality for physics interactions
    - Because all object data is resident on the GPU, this API layer is required to allow the CPU program to use the memory
    - Conceptually similar to something like a NoSQL database, though with very strict object structures, rather than open-ended object types.
    - Objects are largely _implied_ structures, there are no literal classes or objects in the GPU layer. All data is stored sequentially in pre-sized arrays. 


- [integrate.cl](https://github.com/controllerface/bvge/blob/main/src/main/resources/cl/kernels/integrate.cl): Equations of Motion
  - All physics simulations must perform some logic that uses the equations of motion to determine where all the objects are in the current frame. These calculations take into account position, velocity, acceleration, and other forces. 
  - This physics simulation uses an integration algorithm that is based on the [Verlet](https://en.wikipedia.org/wiki/Verlet_integration) integration method. This is a slightly different process than the possibly more well-known Euler method, but it serves the same purpose.
  - Because the integration process involves calculating several properties that are useful for bounding box generation, the kernel does double duty as a [bounding box](https://en.wikipedia.org/wiki/Minimum_bounding_box) generator. This reduces the number of kernel calls needed per frame.


- [aabb_collide.cl](https://github.com/controllerface/bvge/blob/main/src/main/resources/cl/kernels/aabb_collide.cl): Broad-Phase Collision; Axis Aligned Bounding Box (AABB)
  - The first stage of collision detection, this uses AABBs calculated during integration, to do a rough spatial check between objects for collision
  - Less computationally expensive than narrow-phase, helps reduce the number of objects that would be processed in that later, more expensive step.
  - Can be thought of as a check to see if two objects are "close enough" to require further inspection to see if they _really_ touch.


- [sat_collide.cl](https://github.com/controllerface/bvge/blob/main/src/main/resources/cl/kernels/sat_collide.cl): Narrow-Phase Collision; Separating-Axis Theorem
  - Objects that are found to have AABB collisions are allowed to be further processed to determine if they actually do overlap in physical space
  - Uses the well studied and widely used [Separating-Axis Theorem](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem) (which is grouped into the broader concept of Hyperplane separation) though in this case we are only concerned with the separating axes.  
  - Exposes extra kernels that help apply reactions to objects that are found to be colliding. 
  - Unlike many common implementations, this process does not require collision manifolds. Instead, there is a point-aligned memory buffer that is used to store accumulated reactions on each point
  - Because the same point may be affected in more than one collision, care is taken to ensure reaction vectors are applied _cumulatively_.  

Current Status
-
Currently, the engine supports creating, reading, and updating of all required objects. There are several debugging renderers and one texture renderer implemented. The last main hurdle remains support for deleting of objects.

Toward implementing deleting objects, I have made some progress both on the Open CL kernels required to perform the buffer scan and compaction, and the changes to the renderers that will be needed.

I refactored the edge renderer to be able to handle deletes and took the opportunity to clean it up, so it is quite a bit less code now. This is a very good example of how the memory layout can be employed to make a simple and efficient renderer in this prototype engine.

In doing this process, I did find an issue with models that makes the current design cumbersome, so I have started in on a refactor to the circle renderer, which will serve as a basis for more complex models. The new design will work very similarly to how it functions today, accept the `indices` value that is commonly used by the render batches will be computed rather than taken from the `Models` class.

This new design required extending the armature flag data to `int2` with the current root hull ID stored in `x` and the model ID in `y`. This will allow CL kernels to easily scan the buffers for instances of a specific model to render, the ultimate destination being a GL buffer that is used to render the models using instanced rendering.

The intent is to only store a raw model count in the Models class, instead of the current design where a root hull or armature ID is mapped to each instance. The current model renderers rely on this, which is why they need to be updated. Moving to this new design also makes it possible to do the final stage of the delete process, which can be reduced to simple counter decrement instead of needing to remove a value in a mapped list.  