import numpy as np

from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class Ice(WorldObj):
    def __init__(self):
        super().__init__('ice', 'blue')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (119, 201, 240)  # Pale blue

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)


# Add Ice top object index.
OBJECT_TO_IDX['ice'] = max(OBJECT_TO_IDX.values()) + 1


class IceGridEnv(MiniGridEnv):
    def __init__(self, size, mass=1, friction_norm=0.8):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            see_through_walls=False,
            seed=None
        )

        self.mass = mass
        self.friction_norm = friction_norm
        self.agent_velocity = np.array([0, 0])

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        self.grid = Grid(width, height)

        # Surrounding walls.
        self.grid.wall_rect(0, 0, width, height)

        # Fill with ice.
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                self.put_obj(Ice(), i, j)

        # Agent top left.
        self.agent_pos = [1, 1]
        self.agent_dir = 0

        # Place goal bottom right.
        self.goal_pos = [
            self._rand_int(5, width - 1),
            self._rand_int(5, height - 1)
        ]
        self.put_obj(Goal(), *self.goal_pos)

        self.mission = "Get to the goal square"

    def take_discrete_step(self):
        """Take one time step wrt the velocity vector."""
        self.agent_pos = np.round(
            self.agent_pos + self.agent_velocity
        ).astype(int)

        if self.agent_pos[0] < 1:
            self.agent_pos[0] = 1
            self.agent_velocity[0] = 0

        elif self.agent_pos[0] > self.grid.width - 1:
            self.agent_pos[0] = self.grid.width - 2
            self.agent_velocity[0] = 0

        if self.agent_pos[1] < 1:
            self.agent_pos[1] = 1
            self.agent_velocity[1] = 0

        elif self.agent_pos[1] > self.grid.height - 1:
            self.agent_pos[1] = self.grid.height - 2
            self.agent_velocity[1] = 0

    def step(self, action):
        if action != self.actions.forward:
            return super().step(action)

        # Move with inertia.
        # Force applied by action.
        # The constant here to determine how strong the agent gets pushed.
        force = 2.0 * DIR_TO_VEC[self.agent_dir]

        # Add friction.
        if self.agent_velocity.sum() > 0:
            friction = (
                -self.friction_norm *
                self.agent_velocity / np.linalg.norm(self.agent_velocity)
            )
            force += friction

        acceleration = force / self.mass
        self.agent_velocity = self.agent_velocity + acceleration

        # Discretize the updated square.
        self.take_discrete_step()
        cell = self.grid.get(*self.agent_pos)
        reward = 0
        done = False

        if cell != None and cell.type == 'goal':
            done = True
            reward = self._reward()

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}



class IceGridS50Env(IceGridEnv):
    def __init__(self):
        super().__init__(size=50)


register(
    id='MiniGrid-IceGridS50-v0',
    entry_point='ice_slippery:IceGridS50Env'
)