from enum import Enum
import numpy as np
import os
import json
from PIL import Image


class Env:
    channels = {
    }
    action_map = ['nop', 'left', 'up', 'right', 'down', 'jump']
    level_name = 'level'

    # This signature is required by the Environment class, although ramping is not used here.
    def __init__(self, ramping=None, random_state=None):
        self.channels = Env.channels
        self.random = np.random.RandomState() if random_state is None else random_state
        self.screen_size = Level.level_size
        self.screen_state = None
        self.level = Level.load_level(Env.level_name)
        self.reset()

    def reset(self):
        pass
        # TODO

    def act(self, action_num):
        # TODO
        # return (reward, terminated?)
        return 0, False

    def state_shape(self):
        return *self.screen_size, len(self.channels)

    def state(self):
        return self.screen_state

    def _create_screen_state(self):
        screen_state = np.zeros(self.state_shape(), dtype=bool)
        # TODO
        # screen_state[0, :, Env.channels['gauge_background']] = True
        return screen_state

    def _update_screen_state(self, state):
        pass
        # TODO

    def handle_human_action(self, action):
        pass


class Level:
    level_size = (20, 13)  # width, height

    def __init__(self, layout, enemies_starts, player_start):
        self.layout, self.enemies_starts, self.player_start = layout, enemies_starts, player_start

    @staticmethod
    def load_level(level_name):
        file_name = _get_file_location(level_name + '.json')
        with open(file_name, 'r') as level_file:
            descriptor_data = json.load(level_file)
            assert 'layout_file' in descriptor_data, 'level descriptor file must specify level layout file'
            layout_file = _get_file_location(descriptor_data['layout_file'])
            layout = Level._load_level_layout(layout_file)
            enemies_starts = map(Level._position_array_to_tuple, descriptor_data['enemies_starts'])
            player_start = Level._position_array_to_tuple(descriptor_data['player_start'])
            return Level(layout, enemies_starts, player_start)

    @staticmethod
    def _load_level_layout(layout_file):
        assert layout_file[-4:] == '.png'
        layout_image = Image.open(layout_file)
        layout_pixels = layout_image.load()
        assert layout_image.size == Level.level_size
        w, h = layout_image.size
        return [[_color_to_tile[layout_pixels[col, row]] for col in range(w)] for row in range(h)]

    @staticmethod
    def _position_array_to_tuple(position_array):
        assert len(position_array) == 2
        return position_array[0], position_array[1]


def _get_file_location(file_name):
    return os.path.join('data', 'pacman', file_name)


class LevelTile(Enum):
    empty = 0
    left_right = 1
    up_down = 2
    down_right = 3
    up_right = 4
    left_up = 5
    left_down = 6
    left_up_right = 7
    up_right_down = 8
    right_down_left = 9
    down_left_up = 10
    left_up_right_down = 11


# _tile_to_channel = {
#     LevelTile.empty: 'wall',
# }


def _hex(hexcode):
    return (hexcode >> 16) & 0xff, (hexcode >> 8) & 0xff, hexcode & 0xff


_color_to_tile = {
    _hex(0xffffff): LevelTile.empty,
    _hex(0x64510b): LevelTile.left_right,
    _hex(0x000000): LevelTile.up_down,
    _hex(0x00ff00): LevelTile.down_right,
    _hex(0x0000ff): LevelTile.up_right,
    _hex(0x00ffff): LevelTile.left_up,
    _hex(0xff00ff): LevelTile.left_down,
    _hex(0xff7e70): LevelTile.left_up_right,
    _hex(0x706c70): LevelTile.up_right_down,
    _hex(0xffff00): LevelTile.right_down_left,
    _hex(0x007ea1): LevelTile.down_left_up,
    _hex(0xff0000): LevelTile.left_up_right_down,
}
