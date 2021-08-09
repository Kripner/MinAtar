from enum import Enum
from collections import namedtuple
import numpy as np
import os
import json
from PIL import Image
import math


class Env:
    channels = {
        'wall': 0,
        'disappearing_wall': 1,
        'moving_sand': 2,
        'door': 3,
        'laser_door': 4,
        'gauge_background': 5,
        'gauge_health': 6,
        'gauge_keys': 7,
        'lava': 8,
        'ladder': 9,
        'player': 10,
        'enemy': 11,
        'key': 12
    }
    action_map = ['nop', 'left', 'up', 'right', 'down', 'jump']
    walking_speed = 1
    lateral_jumping_speed = 0.5
    ladder_speed = 0.5
    moving_sand_speed = 0.5
    gravity = 0.3
    jump_force = 1
    initial_room = 'room-35'  # TODO: change

    # This signature is required by the Environment class, although ramping is not used here.
    def __init__(self, ramping=None, random_state=None):
        self.channels = Env.channels
        self.random = np.random.RandomState() if random_state is None else random_state
        self.screen_size = (Room.room_size[0], Room.room_size[1] + 1)  # one row contains the HUD
        # For documentation purposes, overridden in reset().
        self.rooms, self.room, self.soft_reset_position, self.screen_state, self.human_checkpoint = \
            None, None, None, None, None
        self.player = Player(self)
        self.reset()

    def reset(self):
        self.rooms = RoomCache()
        self._change_room(Env.initial_room)
        assert self.room.player_start is not None, 'Initial room does not specify initial player position.'
        self.soft_reset_position = self.room.player_start
        self.player.reset()
        self.screen_state = self._create_state()

    # called after player loses one hearth
    def _soft_reset(self):
        self.player.soft_reset()
        self.screen_state = self._create_state()

    def act(self, action_num):
        action = Env.action_map[action_num]
        self.player.update(action)
        self.room.update(self.player)
        if self.player.dead or self._has_collided():
            if self.player.health == 0:
                self._update_state(self.screen_state)  # get the last frame
                return 0, True  # player died
            self.player.health -= 1
            self._soft_reset()
        self._update_state(self.screen_state)
        return 0, False  # reward, terminated

    def _has_collided(self):
        player_cell = Env.position_to_cell(self.player.player_pos)
        if self.room.at(player_cell) == RoomTile.lava:
            return True
        if self.room.at_moving(player_cell) == MovingObject.enemy:
            return True

    def state_shape(self):
        return *self.screen_size, len(self.channels)

    def state(self):
        return self.screen_state

    def _create_state(self):
        screen_state = np.zeros(self.state_shape(), dtype=bool)
        screen_state[0, :, self.channels['gauge_background']] = True
        lvl = self.room.room_data
        for y in range(Room.room_size[1]):
            for x in range(Room.room_size[0]):
                tile = lvl[y][x]
                if tile in _tile_to_channel:
                    screen_state[y + 1, x, self.channels[_tile_to_channel[tile]]] = True
        return screen_state

    def _update_state(self, state):
        Player.reset_state(state)
        self.player.add_to_state(state)

        width, height = self.screen_size
        state[0, :, self.channels['gauge_health']] = False
        state[0, 0:self.player.health, self.channels['gauge_health']] = True
        state[0, :, self.channels['gauge_keys']] = False
        state[0, width - self.player.key_count:width, self.channels['gauge_keys']] = True

        self.room.update_state(state)
        return state

    def try_changing_room(self, new_player_cell):
        y, x = new_player_cell
        next_neighbour = None
        next_location = None
        if x < 0:
            next_neighbour = 'left_neighbour'
            next_location = (y, Room.room_size[0] - 1)
        if x >= Room.room_size[0]:
            next_neighbour = 'right_neighbour'
            next_location = (y, 0)
        if y < 0:
            next_neighbour = 'top_neighbour'
            next_location = (Room.room_size[1] - 1, x)
        if y >= Room.room_size[1]:
            next_neighbour = 'bottom_neighbour'
            next_location = (0, x)

        if next_neighbour in self.room.neighbours:
            self._change_room(self.room.neighbours[next_neighbour])
            self._jump_to_cell(next_location)
            self.soft_reset_position = next_location
            self.screen_state = self._create_state()
            return True
        return False

    def _change_room(self, lvl_name):
        self.room = self.rooms.get_room(lvl_name)

    def _jump_to_cell(self, cell):
        self.player.player_pos = Env.cell_to_position(cell)

    @staticmethod
    def position_to_cell(position):
        y, x = position
        return math.floor(y), math.floor(x)

    @staticmethod
    def cell_to_position(cell):
        return cell

    def _get_checkpoint(self):
        return Checkpoint(
            room_name=self.room.room_name,
            soft_reset_position=self.soft_reset_position,
            player_pos=self.player.player_pos,
            player_speed=self.player.player_speed,
            player_state=self.player.player_state,
            exiting_ladder=self.player.exiting_ladder
        )

    def _apply_checkpoint(self, checkpoint):
        self._change_room(checkpoint.room_name)
        self.soft_reset_position = checkpoint.soft_reset_position
        self.player.player_pos = checkpoint.player_pos
        self.player.player_speed = checkpoint.player_speed
        self.player.player_state = checkpoint.player_state
        self.player.exiting_ladder = checkpoint.exiting_ladder
        self.screen_state = self._create_state()

    def handle_human_action(self, action):
        action = action.lower()
        if action == 's':  # save checkpoint
            self.human_checkpoint = self._get_checkpoint()
        elif action == 'l':  # load checkpoint
            if self.human_checkpoint is not None:
                self._apply_checkpoint(self.human_checkpoint)


Checkpoint = namedtuple('Checkpoint',
                        ['room_name', 'soft_reset_position', 'player_pos', 'player_speed',
                         'player_state', 'exiting_ladder'])


class PlayerState(Enum):
    standing = 0
    flying = 1
    on_ladder = 2


class Room:
    room_size = (20, 19)  # (width, height)
    neighbours_names = ['left_neighbour', 'top_neighbour', 'right_neighbour', 'bottom_neighbour']

    def __init__(self, room_name, room_data, neighbours):
        self.room_name = room_name
        self.room_data = room_data
        self.neighbours = neighbours
        player_starts = [(y, x)
                         for y in range(Room.room_size[1])
                         for x in range(Room.room_size[0])
                         if room_data[y][x] == RoomTile.player_start]
        assert len(player_starts) <= 1
        self.player_start = player_starts[0] if len(player_starts) == 1 else None

        enemies_starts = [(y, x)
                          for y in range(Room.room_size[1])
                          for x in range(Room.room_size[0])
                          if room_data[y][x] == RoomTile.enemy_start]
        enemies = [Enemy(cell, self) for cell in enemies_starts]

        # Doors are supposed to be vertical bars. This detects the top of each such bar.
        door_tops = [(y, x)
                     for y in range(Room.room_size[1])
                     for x in range(Room.room_size[0])
                     if room_data[y][x] == RoomTile.door and (
                             y - 1 < 0 or room_data[y - 1][x] != RoomTile.door)]
        doors = [Door(cell, self) for cell in door_tops]

        keys_positions = [(y, x)
                          for y in range(Room.room_size[1])
                          for x in range(Room.room_size[0])
                          if room_data[y][x] == RoomTile.key]
        keys = [Key(cell, self) for cell in keys_positions]

        laser_door_tops = [(y, x)
                           for y in range(Room.room_size[1])
                           for x in range(Room.room_size[0])
                           if room_data[y][x] == RoomTile.laser_door and (
                                   y - 1 < 0 or room_data[y - 1][x] != RoomTile.laser_door)]
        laser_doors = [LaserDoor(cell, self) for cell in laser_door_tops]

        disappearing_walls = [DisappearingWall((y, x), self)
                              for y in range(Room.room_size[1])
                              for x in range(Room.room_size[0])
                              if room_data[y][x] == RoomTile.disappearing_wall]

        self.moving_parts = enemies + doors + keys + laser_doors + disappearing_walls

        # Just for documentation, overridden in _update_moving_state.
        self.moving_data = None
        self._update_moving_state()

    def update(self, player):
        for o in self.moving_parts:
            o.update(player)
        self._update_moving_state()

    def _update_moving_state(self):
        moving_data = [[MovingObject.none
                        for _ in range(Room.room_size[0])]
                       for _ in range(Room.room_size[1])]
        for o in self.moving_parts:
            o.draw(moving_data)
        self.moving_data = moving_data

    def update_state(self, state):
        Enemy.reset_state(state)
        Door.reset_state(state)
        Key.reset_state(state)
        LaserDoor.reset_state(state)
        DisappearingWall.reset_state(state)

        for o in self.moving_parts:
            o.add_to_state(state)

    def reset(self):
        for o in self.moving_parts:
            o.reset()

    @staticmethod
    def is_inside(cell):
        y, x = cell
        return 0 <= x < Room.room_size[0] and 0 <= y < Room.room_size[1]

    def at(self, cell):
        if not Room.is_inside(cell):
            return RoomTile.wall
        return self.room_data[cell[0]][cell[1]]

    def at_moving(self, cell):
        if not Room.is_inside(cell):
            return RoomTile.empty
        return self.moving_data[cell[0]][cell[1]]

    def is_solid_at(self, cell):
        return self.at(cell) in [RoomTile.wall, RoomTile.moving_sand] or self.at_moving(cell) in [MovingObject.door]


class Door:
    def __init__(self, top_position, room):
        y, x = top_position
        height = 1
        while room.at((y + height, x)) == RoomTile.door:
            height += 1

        self.top_position = top_position
        self.height = height
        self.open = False

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['door']] = False

    def add_to_state(self, state):
        if self.open:
            return
        top_y, top_x = self.top_position
        state[top_y + 1:top_y + self.height + 1, top_x, Env.channels['door']] = True

    def draw(self, moving_data):
        if self.open:
            return
        top_y, top_x = self.top_position
        for y in range(top_y, top_y + self.height):
            moving_data[y][top_x] = MovingObject.door

    def update(self, player):
        if self.open or player.key_count == 0:
            return
        top_y, top_x = self.top_position
        player_y, player_x = player.get_player_cell()
        if abs(player_x - top_x) <= 1 and top_y <= player_y < top_y + self.height:
            self.open = True
            player.key_count -= 1


class Key:
    def __init__(self, position, room):
        self.position = position
        self.collected = False

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['key']] = False

    def add_to_state(self, state):
        if self.collected:
            return
        y, x = self.position
        state[y + 1, x, Env.channels['key']] = True

    def draw(self, moving_data):
        if self.collected:
            return
        y, x = self.position
        moving_data[y][x] = MovingObject.key

    def update(self, player):
        if self.collected:
            return
        if player.get_player_cell() == self.position:
            self.collected = True
            player.key_count += 1


class LaserDoor:
    opened_duration = 5
    closed_duration = 10

    def __init__(self, top_position, room):
        y, x = top_position
        height = 1
        while room.at((y + height, x)) == RoomTile.laser_door:
            height += 1

        self.top_position = top_position
        self.height = height
        self.open = False
        self.ticks_since_switched = 0

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['laser_door']] = False

    def add_to_state(self, state):
        if self.open:
            return
        top_y, top_x = self.top_position
        state[top_y + 1:top_y + self.height + 1, top_x, Env.channels['laser_door']] = True

    def draw(self, moving_data):
        if self.open:
            return
        top_y, top_x = self.top_position
        for y in range(top_y, top_y + self.height):
            moving_data[y][top_x] = MovingObject.laser_door

    def update(self, player):
        self.ticks_since_switched += 1
        should_switch = self.ticks_since_switched == (
            LaserDoor.opened_duration if self.open else LaserDoor.closed_duration)
        if should_switch:
            self.open = not self.open
            self.ticks_since_switched = 0

        if self.open:
            return
        top_y, top_x = self.top_position
        player_y, player_x = player.get_player_cell()
        if player_x == top_x and top_y <= player_y < top_y + self.height:
            player.die()


class DisappearingWall:
    present_duration = 10
    absent_duration = 5

    def __init__(self, position, room):
        self.position = position
        self.present = False
        self.ticks_since_switched = 0

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['disappearing_wall']] = False

    def add_to_state(self, state):
        if not self.present:
            return
        y, x = self.position
        state[y + 1, x, Env.channels['disappearing_wall']] = True

    def draw(self, moving_data):
        if not self.present:
            return
        y, x = self.position
        moving_data[y][x] = MovingObject.disappearing_wall

    def update(self, player):
        self.ticks_since_switched += 1
        should_switch = self.ticks_since_switched >= (
            DisappearingWall.present_duration if self.present else DisappearingWall.absent_duration)
        if should_switch:
            self.present = not self.present
            self.ticks_since_switched = 0


class Player:
    _max_hearths = 5

    def __init__(self, environment):
        self.environment = environment
        # For documentation purposes, overridden in reset().
        self.player_speed, self.player_pos, self.player_state, self.exiting_ladder, self.health, self.dead, self.key_count = \
            None, None, None, None, None, None, None

    def reset(self):
        self.soft_reset()
        self.health = Player._max_hearths
        self.key_count = 5  # TODO: change

    # called after player loses one hearth
    def soft_reset(self):
        self.player_pos = Env.cell_to_position(self.environment.soft_reset_position)
        self.player_speed = np.array([0, 0], dtype=np.float32)
        self.player_state = PlayerState.standing
        self.exiting_ladder = False
        self.dead = False

    def get_player_cell(self):
        return Env.position_to_cell(self.player_pos)

    def die(self):
        self.dead = True

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['player']] = False

    def add_to_state(self, state):
        player_cell = Env.position_to_cell(self.player_pos)
        state[player_cell[0] + 1, player_cell[1], Env.channels['player']] = True

    def update(self, action):
        y, x = self.get_player_cell()
        new_player_pos = self._calculate_new_position(action)
        new_player_cell = Env.position_to_cell(new_player_pos)
        new_y, new_x = new_player_cell

        crashed = False
        changed_room = False
        if not self.environment.room.is_inside(new_player_cell):
            changed_room = self.environment.try_changing_room(new_player_cell)
            crashed = not changed_room
        elif not (new_x == x and new_y < y) and \
                not self.environment.room.is_solid_at((y, x)) and \
                self.environment.room.is_solid_at(new_player_cell):
            crashed = True

        if not crashed and not changed_room:
            self.player_pos = new_player_pos

        if crashed:
            self.player_speed = np.array([0, 0], dtype=np.float32)

    def _get_new_player_state(self):
        player_cell = self.get_player_cell()
        cell_bellow = (player_cell[0] + 1, player_cell[1])

        can_be_on_ladder = self.environment.room.at(player_cell) == RoomTile.ladder
        can_stand = (not Room.is_inside(cell_bellow)) or \
                    self.environment.room.at(cell_bellow) in [RoomTile.wall, RoomTile.moving_sand] or \
                    (self.environment.room.at(cell_bellow) == RoomTile.ladder and not can_be_on_ladder) or \
                    self.environment.room.at_moving(cell_bellow) in [MovingObject.disappearing_wall, MovingObject.door]

        if self.player_state == PlayerState.flying:
            if can_be_on_ladder and not self.exiting_ladder:
                return PlayerState.on_ladder
            if can_stand:
                return PlayerState.standing
        if self.player_state == PlayerState.standing:
            if not can_stand:
                return PlayerState.flying
        if self.player_state == PlayerState.on_ladder:
            if self.exiting_ladder or not can_be_on_ladder:
                self.exiting_ladder = False
                return PlayerState.standing if can_stand else PlayerState.flying

        if not can_be_on_ladder:
            self.exiting_ladder = False
        return self.player_state

    # Calculates new position of the player not counting in collisions.
    def _calculate_new_position(self, action):
        # Player's position has floating-point coordinates that get floored when displaying. Physics behaves as if the
        # player was a single point.
        self.player_state = self._get_new_player_state()
        player_cell = self.get_player_cell()
        cell_bellow = (player_cell[0] + 1, player_cell[1])
        standing, on_ladder, flying = self._one_hot_state()

        ladder_exiting_action = False
        if standing or on_ladder:
            self.player_speed[0] = 0
            if action == 'jump':
                self.player_speed[0] -= Env.jump_force
                ladder_exiting_action = True
                self.player_state = PlayerState.flying

        if flying:
            self.player_speed[0] += Env.gravity

        new_player_pos = self.player_pos + self.player_speed
        if standing or on_ladder:
            if action == 'left':
                new_player_pos[1] -= Env.walking_speed
                ladder_exiting_action = True
            elif action == 'right':
                new_player_pos[1] += Env.walking_speed
                ladder_exiting_action = True

        if flying:
            if action == 'left':
                new_player_pos[1] -= Env.lateral_jumping_speed
            elif action == 'right':
                new_player_pos[1] += Env.lateral_jumping_speed

        if on_ladder:
            if ladder_exiting_action:
                self.exiting_ladder = True
            else:
                if action == 'up':
                    new_player_pos[0] -= Env.ladder_speed
                elif action == 'down':
                    new_player_pos[0] += Env.ladder_speed

        if standing:
            if self.environment.room.at(cell_bellow) == RoomTile.ladder and action == 'down':
                new_player_pos[0] += Env.ladder_speed
                self.player_state = PlayerState.on_ladder
            elif self.environment.room.at(player_cell) == RoomTile.ladder and action == 'up':
                new_player_pos[0] -= Env.ladder_speed
                self.player_state = PlayerState.on_ladder

            if self.environment.room.at(cell_bellow) == RoomTile.moving_sand:
                new_player_pos[1] -= Env.moving_sand_speed

        return new_player_pos

    def _one_hot_state(self):
        return self.player_state == PlayerState.standing, self.player_state == PlayerState.on_ladder, \
               self.player_state == PlayerState.flying


class Enemy:
    _ticks_per_move = 2  # inverse of the speed of an enemy

    def __init__(self, starting_cell, room):
        self.starting_cell = starting_cell
        self.room = room
        # For documentation purposes, overridden in reset().
        self.enemy_cell, self.ticks_since_move, self.previous_cell = None, None, None
        self.reset()

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['enemy']] = False

    def add_to_state(self, state):
        state[self.enemy_cell[0] + 1, self.enemy_cell[1], Env.channels['enemy']] = True

    def draw(self, moving_data):
        y, x = self.enemy_cell
        moving_data[y][x] = MovingObject.enemy

    def update(self, player):
        if self.ticks_since_move + 1 == Enemy._ticks_per_move:
            self._move()
            self.ticks_since_move = 0
        else:
            self.ticks_since_move += 1

    def _move(self):
        path_neighbours = [c for c in neighbour_cells(self.enemy_cell) if
                           self.room.at(c) == RoomTile.enemy_path or self.room.at(c) == RoomTile.enemy_start]
        assert len(path_neighbours) <= 2
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
    if y != Room.room_size[1] - 1:
        yield y + 1, x
    if x != 0:
        yield y, x - 1
    if x != Room.room_size[0] - 1:
        yield y, x + 1


class RoomCache:
    def __init__(self):
        self.rooms = dict()

    def get_room(self, room_name):
        if room_name not in self.rooms:
            self.rooms[room_name] = RoomCache._load_room(room_name)
        return self.rooms[room_name]

    @staticmethod
    def _load_room(room_name):
        file_name = _get_file_location(room_name + '.json')
        with open(file_name, 'r') as room_file:
            data = json.load(room_file)
            room_data = RoomCache._load_room_data(_get_file_location(data['data_file']))
            neighbours = {neighbour: data[neighbour] for neighbour in Room.neighbours_names if
                          neighbour in data}
            return Room(room_name, room_data, neighbours)

    @staticmethod
    def _load_room_data(data_file):
        assert data_file[-4:] == '.png'
        data_image = Image.open(data_file)
        data_pixels = data_image.load()
        assert data_image.size == Room.room_size
        w, h = data_image.size
        return [[_color_to_tile[data_pixels[x, y]] for x in range(w)] for y in range(h)]


def _get_file_location(file_name):
    return os.path.join('data/montezumas-revenge', file_name)


class MovingObject(Enum):
    none = 0
    enemy = 1
    door = 2
    key = 3
    laser_door = 4
    disappearing_wall = 5


class RoomTile(Enum):
    empty = 0
    wall = 1
    player_start = 2
    lava = 3
    ladder = 4
    enemy_path = 5
    enemy_start = 6
    moving_sand = 7
    door = 8
    key = 9
    laser_door = 10
    disappearing_wall = 11
    coin = 12


_tile_to_channel = {
    RoomTile.wall: 'wall',
    RoomTile.lava: 'lava',
    RoomTile.ladder: 'ladder',
    RoomTile.moving_sand: 'moving_sand'
}


def _hex(hexcode):
    return (hexcode >> 16) & 0xff, (hexcode >> 8) & 0xff, hexcode & 0xff


_color_to_tile = {
    _hex(0xffffff): RoomTile.empty,
    _hex(0x000000): RoomTile.wall,
    _hex(0x00ff00): RoomTile.player_start,
    _hex(0xff0000): RoomTile.lava,
    _hex(0xffff00): RoomTile.ladder,
    _hex(0x00ffff): RoomTile.enemy_path,
    _hex(0x0000ff): RoomTile.enemy_start,
    _hex(0xff00ff): RoomTile.moving_sand,
    _hex(0xac6000): RoomTile.door,
    _hex(0xac8f00): RoomTile.key,
    _hex(0x584052): RoomTile.laser_door,
    _hex(0x626262): RoomTile.disappearing_wall,
    _hex(0xff9400): RoomTile.coin
}
