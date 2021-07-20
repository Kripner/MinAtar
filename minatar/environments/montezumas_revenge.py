from enum import Enum
import numpy as np
import os
import json
from PIL import Image
import math


class Env:
    channels = {
        'wall': 0,
        'gauge': 1,
        'lava': 2,
        'ladder': 3,
        'player': 4,
        'enemy': 5
    }
    action_map = ['nop', 'left', 'up', 'right', 'down', 'jump']
    walking_speed = 1
    gravity = 0.3
    jump_force = 1

    # This signature is required by the Environment class, although ramping is not used here.
    def __init__(self, ramping=None, random_state=None):
        self.channels = Env.channels
        self.random = np.random.RandomState() if random_state is None else random_state
        self.screen_size = (Level.room_size[0], Level.room_size[1] + 1)  # one row contains the HUD
        # For documentation purposes, overridden in reset().
        self.levels, self.level, self.screen_state = None, None, None
        self.player = Player(self)
        self.reset()

    def reset(self):
        self.levels = LevelCache()
        self._change_level('lvl-0')
        self.player.reset()
        self.screen_state = self._create_state()

    def act(self, action_num):
        action = Env.action_map[action_num]
        self.player.update(action)
        self.level.update()
        self.screen_state = self._update_state(self.screen_state)
        if self._has_collided():
            return 0, True  # player died
        return 0, False  # reward, terminated

    def _has_collided(self):
        player_cell = Env.position_to_cell(self.player_pos)
        if self.level.at(player_cell) == LevelTile.lava:
            return True
        for e in self.level.enemies:
            if e.enemy_cell == player_cell:
                return True

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
                if tile in _tile_to_channel:
                    screen_state[y + 1, x, self.channels[_tile_to_channel[tile]]] = True
        player_cell = Env.position_to_cell(self.player_pos)
        screen_state[player_cell[0] + 1, player_cell[1], self.channels['player']] = True
        return screen_state

    def _update_state(self, state):
        state[:, :, self.channels['player']] = False
        player_cell = Env.position_to_cell(self.player_pos)
        state[player_cell[0] + 1, player_cell[1], self.channels['player']] = True

        state[:, :, self.channels['enemy']] = False
        for e in self.level.enemies:
            state[e.enemy_cell[0] + 1, e.enemy_cell[1], self.channels['enemy']] = True
        return state

    def try_changing_level(self, new_player_cell):
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
            self._change_level(self.level.neighbours[next_neighbour])
            self._jump_to_cell(next_location)
            self.screen_state = self._create_state()
            return True
        return False

    def _change_level(self, lvl_name):
        self.level = self.levels.get_level(lvl_name)

    def _jump_to_cell(self, cell):
        self.player_pos = Env.cell_to_position(cell)

    @staticmethod
    def position_to_cell(position):
        y, x = position
        return math.floor(y), math.floor(x)

    @staticmethod
    def cell_to_position(cell):
        return cell


class PlayerState(Enum):
    standing = 0
    flying = 1
    on_ladder = 2


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
        assert (len(player_starts) <= 1)
        self.player_start = player_starts[0] if len(player_starts) == 1 else None

        enemies_starts = [(y, x)
                          for y in range(Level.room_size[1])
                          for x in range(Level.room_size[0])
                          if room_data[y][x] == LevelTile.enemy_start]
        self.enemies = [Enemy(cell, self) for cell in enemies_starts]

    def update(self):
        for e in self.enemies:
            e.update()

    def reset(self):
        for e in self.enemies:
            e.reset()

    @staticmethod
    def is_inside(cell):
        y, x = cell
        return 0 <= x < Level.room_size[0] and 0 <= y < Level.room_size[1]

    def is_full(self, cell):
        return (not self.is_inside(cell)) or self.at(cell) == LevelTile.wall

    def at(self, position):
        return self.room_data[position[0]][position[1]]


class Player:
    def __init__(self, environment):
        self.environment = environment
        # For documentation purposes, overridden in reset().
        self.player_speed, self.player_pos, self.player_state, self.exiting_ladder = \
            None, None, None, None
        self.reset()

    def reset(self):
        self.player_pos = Env.cell_to_position(self.environment.level.player_start)
        self.player_speed = np.array([0, 0], dtype=np.float32)
        self.player_state = PlayerState.standing
        self.exiting_ladder = False

    def update(self, action):
        new_player_pos = self._calculate_new_position(action)
        new_player_cell = Env.position_to_cell(new_player_pos)

        crashed = False
        if not self.environment.level.is_inside(new_player_cell):
            crashed = not self.environment.try_changing_level(new_player_cell)
        elif self.environment.level.at(new_player_cell) == LevelTile.wall:
            crashed = True
        else:
            self.player_pos = new_player_pos

        if crashed:
            self.player_speed = np.array([0, 0], dtype=np.float32)

    def _get_new_player_state(self):
        player_cell = Env.position_to_cell(self.player_pos)
        cell_bellow = (player_cell[0] + 1, player_cell[1])
        standing, on_ladder, flying = \
            self.player_state == PlayerState.standing, self.player_state == PlayerState.on_ladder, self.player_state == PlayerState.flying
        if flying and self.environment.level.is_full(cell_bellow):
            return PlayerState.standing
        if standing and not self.environment.level.is_full(cell_bellow):
            return PlayerState.flying
        if on_ladder and self.exiting_ladder:
            return PlayerState.flying
        if self.environment.level.at(player_cell) == LevelTile.ladder:
            if not self.exiting_ladder:
                return PlayerState.on_ladder
        else:
            self.exiting_ladder = False
        return self.player_state

    # Calculates new position of the player not counting in collisions.
    def _calculate_new_position(self, action):
        # Player's position has floating-point coordinates that get floored when displaying. Physics behaves as if the
        # player was a single point.
        self.player_state = self._get_new_player_state()
        standing, on_ladder, flying = \
            self.player_state == PlayerState.standing, self.player_state == PlayerState.on_ladder, self.player_state == PlayerState.flying
        ladder_exiting_action = False
        if standing or on_ladder:
            self.player_speed[0] = 0
            if action == 'jump':
                self.player_speed[0] -= Env.jump_force
                ladder_exiting_action = True

        if flying:
            self.player_speed[0] += Env.gravity

        new_player_pos = self.player_pos + self.player_speed
        if standing or flying or on_ladder:
            if action == 'left':
                new_player_pos[1] -= Env.walking_speed
                ladder_exiting_action = True
            if action == 'right':
                new_player_pos[1] += Env.walking_speed
                ladder_exiting_action = True

        if on_ladder and ladder_exiting_action:
            self.exiting_ladder = True

        return new_player_pos


class Enemy:
    _ticks_per_move = 2  # inverse of the speed of an enemy

    def __init__(self, starting_cell, level):
        self.starting_cell = starting_cell
        self.level = level
        # For documentation purposes, overridden in reset().
        self.enemy_cell, self.ticks_since_move, self.previous_cell = None, None, None
        self.reset()

    def update(self):
        if self.ticks_since_move + 1 == Enemy._ticks_per_move:
            self._move()
            self.ticks_since_move = 0
        else:
            self.ticks_since_move += 1

    def _move(self):
        path_neighbours = [c for c in neighbour_cells(self.enemy_cell) if
                           self.level.at(c) == LevelTile.enemy_path or self.level.at(c) == LevelTile.enemy_start]
        assert (len(path_neighbours) <= 2)
        if len(path_neighbours) == 0:
            next_cell = self.enemy_cell  # stationary enemy
        elif len(path_neighbours) == 1:
            next_cell = path_neighbours[0]
        # len(path_neighbours) == 2
        elif self.previous_cell is None:
            next_cell = path_neighbours[0]  # choose one of the two
        else:
            next_cell = [c for c in path_neighbours if c != self.previous_cell][0]
        self.previous_cell, self.enemy_cell = self.enemy_cell, next_cell

    def reset(self):
        self.enemy_cell = self.starting_cell
        self.ticks_since_move = 0
        self.previous_cell = None


def neighbour_cells(cell):
    y, x = cell
    if y != 0:
        yield y - 1, x
    if y != Level.room_size[1] - 1:
        yield y + 1, x
    if x != 0:
        yield y, x - 1
    if x != Level.room_size[0] - 1:
        yield y, x + 1


class LevelCache:
    def __init__(self):
        self.levels = dict()

    def get_level(self, level_name):
        if level_name not in self.levels:
            self.levels[level_name] = LevelCache._load_level(level_name)
        return self.levels[level_name]

    @staticmethod
    def _load_level(level_name):
        file_name = _get_file_location(level_name + '.json')
        with open(file_name, 'r') as level_file:
            data = json.load(level_file)
            room_data = LevelCache._load_level_data(_get_file_location(data['data_file']))
            neighbours = {neighbour: data[neighbour] for neighbour in Level.neighbours_names if
                          neighbour in data}
            return Level(room_data, neighbours)

    @staticmethod
    def _load_level_data(data_file):
        assert (data_file[-4:] == '.png')
        data_image = Image.open(data_file)
        data_pixels = data_image.load()
        assert (data_image.size == Level.room_size)
        w, h = data_image.size
        return [[_color_to_tile[data_pixels[x, y]] for x in range(w)] for y in range(h)]


def _get_file_location(file_name):
    return os.path.join('data/montezumas-revenge', file_name)


class LevelTile(Enum):
    empty = 0
    wall = 1
    player_start = 2
    lava = 3
    ladder = 4
    enemy_path = 5
    enemy_start = 6


_tile_to_channel = {
    LevelTile.wall: 'wall',
    LevelTile.lava: 'lava',
    LevelTile.ladder: 'ladder'
}

_color_to_tile = {
    (255, 255, 255): LevelTile.empty,
    (0, 0, 0): LevelTile.wall,
    (0, 255, 0): LevelTile.player_start,
    (255, 0, 0): LevelTile.lava,
    (255, 255, 0): LevelTile.ladder,
    (0, 255, 255): LevelTile.enemy_path,
    (0, 0, 255): LevelTile.enemy_start
}
