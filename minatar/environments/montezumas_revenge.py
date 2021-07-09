from enum import Enum
import numpy as np
import os
import json
from PIL import Image
import math


class Env:
    channels = {
        'player': 0,
        'wall': 1,
        'gauge': 2
    }
    action_map = ['nop', 'left', 'up', 'right', 'down', 'jump']
    walking_speed = 1
    gravity = 0.2
    jump_force = 0.8

    def __init__(self, ramping=None, random_state=None):
        self.channels = Env.channels
        self.random = np.random.RandomState() if random_state is None else random_state
        self.screen_size = (Level.room_size[0], Level.room_size[1] + 1)  # one row contains the HUD
        self.reset()

    def act(self, action_num):
        if self.player_speed[0] >= -10e-3:
            self.jumping = False

        action = Env.action_map[action_num]
        new_player_pos = self._handle_movement(action)
        new_player_cell = Env._position_to_cell(new_player_pos)

        crashed = False
        if not self.level.is_inside(new_player_cell):
            crashed = not self._try_changing_level(new_player_cell)
        elif self.level.at(new_player_cell) == LevelTile.wall:
            crashed = True
        else:
            self.player_pos = new_player_pos

        if crashed:
            self.player_speed = np.array([0, 0], dtype=np.float32)

        self._update_state()
        return (0, False)  # reward, terminated

    def _handle_movement(self, action):
        player_cell = Env._position_to_cell(self.player_pos)
        cell_bellow = (player_cell[0] + 1, player_cell[1])
        if self.level.is_full(cell_bellow):
            self.player_speed[0] = 0
            if not self.jumping and action == 'jump':
                self.jumping = True
                self.player_speed[0] -= Env.jump_force
        else:
            self.player_speed[0] += Env.gravity

        new_player_pos = self.player_pos + self.player_speed
        if action == 'left':
            new_player_pos[1] -= Env.walking_speed
        if action == 'right':
            new_player_pos[1] += Env.walking_speed

        return new_player_pos

    def reset(self):
        self._change_level('lvl-0', initial_level=True)
        self.player_speed = np.array([0, 0], dtype=np.float32)
        self.jumping = False

    def state_shape(self):
        return *self.screen_size, len(self.channels)

    def state(self):
        return self.screen_state

    def _create_state(self):
        screen_state = np.zeros(self.state_shape(), dtype=bool)
        screen_state[0, :, self.channels['gauge']] = True
        lvl = self.level.room_data
        for y in range(Level.room_size[1]):
            for x in range(Level.room_size[0]):
                tile = lvl[y][x]
                if tile == LevelTile.empty or tile == LevelTile.player_start:
                    continue
                screen_state[y + 1, x, self.channels[_tile_to_channel[tile]]] = True
        player_cell = Env._position_to_cell(self.player_pos)
        screen_state[player_cell[0] + 1, player_cell[1], self.channels['player']] = True
        return screen_state

    def _update_state(self):
        self.screen_state[:, :, self.channels['player']] = False
        player_cell = Env._position_to_cell(self.player_pos)
        self.screen_state[player_cell[0] + 1, player_cell[1], self.channels['player']] = True

    def _try_changing_level(self, new_player_cell):
        y, x = new_player_cell
        next_neighbour = None
        next_location = None
        if x < 0:
            next_neighbour = 'left-neighbour'
            next_location = (y, Level.room_size[0] - 1)
        if x >= Level.room_size[0]:
            next_neighbour = 'right-neighbour'
            next_location = (y, 0)
        if y < 0:
            next_neighbour = 'top-neighbour'
            next_location = (Level.room_size[1] - 1, x)
        if y >= Level.room_size[1]:
            next_neighbour = 'bottom-neighbour'
            next_location = (0, x)

        if next_neighbour in self.level.neighbours:
            print(next_neighbour)
            self._change_level(self.level.neighbours[next_neighbour])
            self._jump_to_cell(next_location)
            return True
        return False

    def _change_level(self, lvl_name, initial_level=False):
        self.level = load_level(lvl_name)
        if initial_level:
            self.player_pos = Env._cell_to_position(self.level.player_start)
        self.screen_state = self._create_state()

    def _jump_to_cell(self, cell):
        self.player_pos = Env._cell_to_position(cell)

    @staticmethod
    def _position_to_cell(position):
        y, x = position
        return math.floor(y), math.floor(x)

    @staticmethod
    def _cell_to_position(cell):
        return cell


class Level:
    room_size = (20, 19)  # (width, height)
    neighbours_names = ['left-neighbour', 'top-neighbour', 'right-neighbour', 'bottom-neighbour']

    def __init__(self, room_data, neighbours):
        self.room_data = room_data
        self.neighbours = neighbours
        player_starts = [(y, x)
                         for y in range(Level.room_size[1])
                         for x in range(Level.room_size[0])
                         if room_data[y][x] == LevelTile.player_start]
        assert (len(player_starts) == 1)
        self.player_start = player_starts[0]

    def is_inside(self, cell):
        y, x = cell
        return 0 <= x < Level.room_size[0] and 0 <= y < Level.room_size[1]

    def is_full(self, cell):
        return (not self.is_inside(cell)) or self.at(cell) == LevelTile.wall

    def at(self, position):
        return self.room_data[position[0]][position[1]]


class LevelTile(Enum):
    empty = 0
    wall = 1
    player_start = 2


_tile_to_channel = {
    LevelTile.wall: 'wall'
}


def load_level(level_name):
    file_name = get_file_location(level_name + '.json')
    with open(file_name, 'r') as level_file:
        data = json.load(level_file)
        room_data = load_level_data(get_file_location(data['data_file']))
        neighbours = {neighbour: data[neighbour] for neighbour in Level.neighbours_names if neighbour in data}
        return Level(room_data, neighbours)


def get_file_location(file_name):
    return os.path.join('data/montezumas-revenge', file_name)


def load_level_data(data_file):
    assert (data_file[-4:] == '.png')
    data_image = Image.open(data_file)
    data_pixels = data_image.load()
    assert (data_image.size == Level.room_size)
    w, h = data_image.size
    return [[_color_to_tile[data_pixels[x, y]] for x in range(w)] for y in range(h)]


_color_to_tile = {
    (255, 255, 255): LevelTile.empty,
    (0, 0, 0): LevelTile.wall,
    (0, 255, 0): LevelTile.player_start
}
