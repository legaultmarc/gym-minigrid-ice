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
    def __init__(self, size):
        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            see_through_walls=False,
            seed=None
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        self.grid = Grid(width, height)

        # Surrounding walls.
        self.grid.wall_rect(0, 0, width, height)

        # Sample ice patches.
        # Chose top left corner.
        n_patches = 1
        while n_patches > 0:
            patch_width = self._rand_int(2, width - 4)
            patch_height = self._rand_int(2, height - 4)
            # The -2 offset is to account for walls all around the grid.
            patch_top_left = (
                self._rand_int(1, width - patch_width - 2),
                self._rand_int(1, height - patch_height - 2)
            )

            if patch_top_left != (0, 0):
                # Accept patch.
                n_patches -= 1
                self.add_ice_patch(patch_width, patch_height, patch_top_left)

        # Agent top left.
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        # Place goal bottom right.
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        self.mission = "Get to the goal square"

    def add_ice_patch(self, w, h, p):
        for i in range(p[0], p[0] + w):
            for j in range(p[1], p[1] + h):
                self.put_obj(Ice(), i, j)

    @property
    def on_ice(self):
        cur_tile = self.grid.get(*self.agent_pos)
        return cur_tile is not None and cur_tile.type == "ice"

    def step(self, action):
        if not self.on_ice or action != self.actions.forward:
            return super().step(action)

        # Go forward until not on ice.
        while self.on_ice:
            fwd_cell = self.grid.get(*self.front_pos)
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = self.front_pos
            else:
                break

        done = self.step_count >= self.max_steps
        obs = self.gen_obs()
        return obs, 0, done, {}



class IceGridS10Env(IceGridEnv):
    def __init__(self):
        super().__init__(size=10)


register(
    id='MiniGrid-IceGridS10-v0',
    entry_point='ice:IceGridS10Env'
)