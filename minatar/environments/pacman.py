from enum import Enum
import numpy as np
import os
import json
from PIL import Image


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


class Direction(Enum):
    left = 0,
    up = 1,
    right = 2,
    down = 3


_opposite_direction = {
    Direction.left: Direction.right,
    Direction.up: Direction.down,
    Direction.right: Direction.left,
    Direction.down: Direction.up
}

_action_to_direction = {
    'left': Direction.left,
    'up': Direction.up,
    'right': Direction.right,
    'down': Direction.down
}


def _step_in_direction(cell, direction):
    row, col = cell
    if direction == Direction.left:
        return row, col - 1
    if direction == Direction.up:
        return row - 1, col
    if direction == Direction.right:
        return row, col + 1
    if direction == Direction.down:
        return row + 1, col
    assert False


_tile_directions = {
    LevelTile.left_right: {Direction.left, Direction.right},
    LevelTile.up_down: {Direction.up, Direction.down},
    LevelTile.down_right: {Direction.down, Direction.right},
    LevelTile.up_right: {Direction.up, Direction.right},
    LevelTile.left_up: {Direction.left, Direction.up},
    LevelTile.left_down: {Direction.left, Direction.down},
    LevelTile.left_up_right: {Direction.left, Direction.up, Direction.right},
    LevelTile.up_right_down: {Direction.up, Direction.right, Direction.down},
    LevelTile.right_down_left: {Direction.right, Direction.down, Direction.left},
    LevelTile.down_left_up: {Direction.down, Direction.left, Direction.up},
    LevelTile.left_up_right_down: {Direction.left, Direction.up, Direction.right, Direction.down},
}


class WalkingEntity:
    def __init__(self, level, start_cell, ticks_per_move, start_direction):
        self.level, self.start_cell = level, start_cell
        self.ticks_per_move = ticks_per_move
        self.start_direction = start_direction
        # Just for documentation, overridden in reset().
        self.cell, self.previous_cell, self.direction, self.ticks_since_moved = None, None, None, None
        self.reset_walking_entity()

    def reset_walking_entity(self):
        self.cell = self.start_cell
        self.direction = self.start_direction
        self.previous_cell = None
        self.ticks_since_moved = 0

    def update_position(self):
        if self.ticks_since_moved < self.ticks_per_move:
            self.ticks_since_moved += 1
        curr_tile = self.level.at(self.cell)
        if self.ticks_since_moved == self.ticks_per_move and self.direction in _tile_directions[curr_tile]:
            new_cell = _step_in_direction(self.cell, self.direction)
            new_cell_wrapped = WalkingEntity._wrap_cell_around(new_cell)
            if self.level.at(new_cell_wrapped) != LevelTile.empty:
                self.ticks_since_moved = 0
                self.previous_cell, self.cell = self.cell, new_cell_wrapped

    @staticmethod
    def _wrap_cell_around(cell):
        row, col = cell
        if row < 0:
            return Level.level_size[0] - 1, col
        if row == Level.level_size[0]:
            return 0, col
        if col < 0:
            return row, Level.level_size[1] - 1
        if col == Level.level_size[1]:
            return row, 0
        return cell


class Enemy(WalkingEntity):
    def __init__(self, random, level, start_cell):
        super().__init__(level, start_cell, Env.enemy_ticks_per_move, random.choice(Direction))
        self.random = random
        self.dead, self.ticks_since_died = None, None
        self.soft_reset()

    def update(self, player):
        if self.dead:
            self.ticks_since_died += 1
            if self.ticks_since_died == Env.enemy_death_duration:
                self.soft_reset()
            return

        self.direction = self.random.choice(self._get_possible_directions())
        super().update_position()
        self._check_for_collision(player)

    def _get_possible_directions(self):
        curr_tile = self.level.at(self.cell)
        possible_directions, alternative_directions = [], []
        for direction in _tile_directions[curr_tile]:
            new_cell = WalkingEntity._wrap_cell_around(_step_in_direction(self.cell, direction))
            if self.level.at(new_cell) == LevelTile.empty:
                continue
            alternative_directions.append(direction)
            if direction == _opposite_direction[self.direction]:
                continue
            possible_directions.append(direction)
        if len(possible_directions) == 0:
            return alternative_directions
        return possible_directions

    def _check_for_collision(self, player):
        if self.cell == player.cell or {self.cell, self.previous_cell} == {player.cell, player.previous_cell}:
            if player.power_pill_active:
                self.die()
                player.handle_killed_enemy()
            else:
                player.die()

    def die(self):
        self.dead = True

    def soft_reset(self):
        super().reset_walking_entity()
        self.dead = False
        self.ticks_since_died = 0


class Player(WalkingEntity):
    def __init__(self, level, start_cell, start_health):
        super().__init__(level, start_cell, Env.player_ticks_per_move, Direction.right)
        self.health = start_health
        self.score = 0
        self.power_pill_active = False
        self.ticks_since_power_pill_activated = None  # Only meaningful if self.power_pill_active == True.
        self.enemies_killed_by_power_pill = 0
        self.dead = False

    def update(self, action):
        if self.power_pill_active:
            self.ticks_since_power_pill_activated += 1
            if self.ticks_since_power_pill_activated == Env.power_pill_duration:
                self.power_pill_active = False
                self.ticks_since_power_pill_activated = None

        if action in _action_to_direction:
            new_direction = _action_to_direction[action]
            if new_direction in _tile_directions[self.level.at(self.cell)]:
                self.direction = new_direction
        super().update_position()

        if self.level.is_coin_at(self.cell):
            self.level.collect_coin_at(self.cell)
            self.score += Env.score_per_coin
        for power_pill in self.level.power_pills:
            if power_pill == self.cell:
                self.level.power_pills.remove(power_pill)
                self._activate_power_pill()
                break

    def die(self):
        self.dead = True

    def _activate_power_pill(self):
        self.score += Env.power_pill_reward
        if not self.power_pill_active:
            self.enemies_killed_by_power_pill = 0
        self.power_pill_active = True
        self.ticks_since_power_pill_activated = 0

    def handle_killed_enemy(self):
        self.score += Env.enemy_killed_reward[self.enemies_killed_by_power_pill]
        self.enemies_killed_by_power_pill += 1

    def soft_reset(self):
        super().reset_walking_entity()
        self.dead = False


class Level:
    level_size = (13, 20)  # height, width

    def __init__(self, layout, enemies_starts, player_start, power_pills):
        self.layout, self.enemies_starts, self.player_start, self.power_pills = \
            layout, enemies_starts, player_start, power_pills
        self.coins = None
        self._initialize_coins(layout)

    def _initialize_coins(self, layout):
        self.coins = np.zeros(Level.level_size, dtype=np.bool)
        for row in range(Level.level_size[0]):
            for col in range(Level.level_size[1]):
                if layout[row][col] != LevelTile.empty:
                    self.coins[row, col] = True

    def initialize_screen_state(self, screen_state):
        for col in range(Level.level_size[1]):
            for row in range(Level.level_size[0]):
                screen_state[row, col, Env.tile_to_channel[self.layout[row][col]]] = True

    def update_screen_state(self, screen_state):
        for row in range(Level.level_size[0]):
            for col in range(Level.level_size[1]):
                screen_state[row, col, Env.channels['coin']] = self.coins[row, col]

        screen_state[:, :, Env.channels['power_pill']] = False
        for row, col in self.power_pills:
            screen_state[row, col, Env.channels['power_pill']] = True

    def at(self, cell):
        return self.layout[cell[0]][cell[1]]

    def is_coin_at(self, cell):
        return self.coins[cell]

    def collect_coin_at(self, cell):
        self.coins[cell] = False

    @staticmethod
    def load_level(level_name):
        file_name = _get_file_location(level_name + '.json')
        with open(file_name, 'r') as level_file:
            descriptor_data = json.load(level_file)
            assert 'layout_file' in descriptor_data, 'level descriptor file must specify level layout file'
            layout_file = _get_file_location(descriptor_data['layout_file'])
            layout = Level._load_level_layout(layout_file)
            enemies_starts = list(map(Level._position_array_to_tuple, descriptor_data['enemies_starts']))
            player_start = Level._position_array_to_tuple(descriptor_data['player_start'])
            power_pills = list(map(Level._position_array_to_tuple, descriptor_data['power_pills']))
            return Level(layout, enemies_starts, player_start, power_pills)

    @staticmethod
    def _load_level_layout(layout_file):
        assert layout_file[-4:] == '.png'
        layout_image = Image.open(layout_file)
        layout_pixels = layout_image.load()
        assert layout_image.size == (Level.level_size[1], Level.level_size[0])
        w, h = layout_image.size
        return [[_color_to_tile[layout_pixels[col, row]] for col in range(w)] for row in range(h)]

    @staticmethod
    def _position_array_to_tuple(position_array):
        assert len(position_array) == 2
        return position_array[0], position_array[1]


def _get_file_location(file_name):
    return os.path.join('data', 'pacman', file_name)


class Env:
    channels = {
        'coin': 0,
        'power_pill': 1,
        'empty': 2,
        'left_right': 3,
        'up_down': 4,
        'down_right': 5,
        'up_right': 6,
        'left_up': 7,
        'left_down': 8,
        'left_up_right': 9,
        'up_right_down': 10,
        'right_down_left': 11,
        'down_left_up': 12,
        'left_up_right_down': 13,
        'gauge_background': 14,
        'gauge_health': 15,
        'player': 16,
        'enemy': 17
    }
    tile_to_channel = {}
    action_map = ['nop', 'left', 'up', 'right', 'down', 'jump']
    level_name = 'level'
    player_ticks_per_move = 2
    enemy_ticks_per_move = 2
    player_max_health = 2
    score_per_coin = 10
    power_pill_duration = 50
    power_pill_reward = 50
    enemy_killed_reward = [200, 400, 800, 1600]
    enemy_death_duration = 20

    # This signature is required by the Environment class, although ramping is not used here.
    def __init__(self, ramping=None, random_state=None):
        self.channels = Env.channels
        self.random = np.random.RandomState() if random_state is None else random_state
        self.screen_size = (Level.level_size[0] + 1, Level.level_size[1])  # 1 row for the HUD
        # Just for documentation, overridden in reset().
        self.level, self.screen_state, self.player, self.player_pos_inside_cell, self.enemies, self.last_score = \
            None, None, None, None, None, None
        self.reset()

    # Called after player loses one hearth.
    def reset(self):
        self.level = Level.load_level(Env.level_name)
        self.player = Player(self.level, self.level.player_start, Env.player_max_health)
        self.last_score = 0
        self.enemies = []
        for enemy_start in self.level.enemies_starts:
            self.enemies.append(Enemy(self.random, self.level, enemy_start))
        self.screen_state = self._create_screen_state()

    def act(self, action_num):
        action = Env.action_map[action_num]
        self.player.update(action)
        for enemy in self.enemies:
            enemy.update(self.player)
        score_gain = self.player.score - self.last_score
        self.last_score = self.player.score

        if self.player.dead:
            if self.player.health == 0:
                return score_gain, True
            self._soft_reset()
            self.player.health -= 1
        self._update_screen_state(self.screen_state)
        return score_gain, False

    def _soft_reset(self):
        self.player.soft_reset()
        for enemy in self.enemies:
            enemy.soft_reset()

    def state_shape(self):
        return *self.screen_size, len(self.channels)

    def state(self):
        return self.screen_state

    def _create_screen_state(self):
        screen_state = np.zeros(self.state_shape(), dtype=bool)
        self.level.initialize_screen_state(screen_state[1:, :, :])
        screen_state[0, :, Env.channels['gauge_background']] = True
        self._update_screen_state(screen_state)
        return screen_state

    def _update_screen_state(self, screen_state):
        screen_state[:, :, Env.channels['player']] = False
        screen_state[:, :, Env.channels['enemy']] = False
        screen_state[0, :, Env.channels['gauge_health']] = False

        screen_state[0, :self.player.health, Env.channels['gauge_health']] = True
        screen_state[self.player.cell[0] + 1, self.player.cell[1], Env.channels['player']] = True
        for enemy in self.enemies:
            if not enemy.dead:
                screen_state[enemy.cell[0] + 1, enemy.cell[1], Env.channels['enemy']] = True
        self.level.update_screen_state(screen_state[1:, :, :])

    def handle_human_action(self, action):
        pass

    @staticmethod
    def initialize_tile_to_channel():
        Env.tile_to_channel = {}
        for tile in LevelTile:
            Env.tile_to_channel[tile] = Env.channels[tile.name]


Env.initialize_tile_to_channel()


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
