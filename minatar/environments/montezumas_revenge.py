from enum import Enum
from collections import namedtuple
import numpy as np
import os
import json
from PIL import Image
import math


class InventoryItem(Enum):
    torch = 0
    amulet = 1
    key = 2
    sword = 3


class Env:
    channels = {
        'darkness': 0,
        'treasure_room_background': 1,
        'wall': 2,
        'disappearing_wall': 3,
        'moving_sand': 4,
        'door': 5,
        'laser_door': 6,
        'gauge_background': 7,
        'gauge_health': 8,
        'inventory_key': 9,
        'inventory_sword': 10,
        'inventory_torch': 11,
        'inventory_amulet': 12,
        'lava': 13,
        'ladder': 14,
        'player': 15,
        'enemy': 16,
        'key': 17,
        'sword': 18,
        'torch': 19,
        'amulet': 20,
        'coin': 21,
    }
    # TODO: evacuate to a class handling HUD drawing
    _inventory_item_to_gauge_channel = {
        InventoryItem.key: channels['inventory_key'],
        InventoryItem.torch: channels['inventory_torch'],
        InventoryItem.amulet: channels['inventory_amulet'],
        InventoryItem.sword: channels['inventory_sword'],
    }
    action_map = ['nop', 'left', 'up', 'right', 'down', 'jump']
    walking_speed = 1
    lateral_jumping_speed = 0.5
    ladder_speed = 0.8
    moving_sand_speed = 0.5
    gravity = 0.3
    jump_force = 0.9
    initial_room = 'room-35'  # TODO: change
    treasure_room_walk_speed = 1
    score_per_coin = 1000
    amulet_duration = 50

    # This signature is required by the Environment class, although ramping is not used here.
    def __init__(self, ramping=None, random_state=None):
        self.channels = Env.channels
        self.random = np.random.RandomState() if random_state is None else random_state
        self.screen_size = (Room.room_size[0], Room.room_size[1] + 1)  # one row contains the HUD
        self.maze = Maze(self)
        self.treasure_room = None
        self.player = Player()
        # For documentation purposes, overridden in reset().
        self.last_score, self.screen_state, self.human_checkpoint, self.game_state = None, None, None, None
        self.reset()

    def reset(self):
        self._reset_to_starting_point()
        self.player.reset(Env.cell_to_position(self.maze.soft_reset_cell))
        self.trigger_screen_state_redraw()
        self.last_score = self.player.score

    def _reset_to_starting_point(self):
        self.game_state = GameState.in_maze
        self.maze.reset()
        self.player.soft_reset(Env.cell_to_position(self.maze.soft_reset_cell))

    # Called after player loses one hearth.
    def _soft_reset(self):
        self.player.soft_reset(Env.cell_to_position(self.maze.soft_reset_cell))
        self.trigger_screen_state_redraw()

    def act(self, action_num):
        action = Env.action_map[action_num]
        if self.game_state == GameState.in_maze:
            maze_event = self.maze.update(self.player, action)
            if maze_event == MazeEvent.player_died:
                if self.player.health == 0:
                    return self._get_reward(), True  # player died
                self.player.health -= 1
                self._soft_reset()
            elif maze_event == MazeEvent.changed_room:
                self.trigger_screen_state_redraw()
            elif maze_event == MazeEvent.entered_treasure_room:
                self.treasure_room = TreasureRoom(self.random)
                self.game_state = GameState.in_treasure_room
                self.trigger_screen_state_redraw()
            else:
                assert maze_event is None
        elif self.game_state == GameState.in_treasure_room:
            back_to_maze = self.treasure_room.update(self.player, action)
            if back_to_maze:
                self._reset_to_starting_point()
                self.trigger_screen_state_redraw()
        else:
            assert False
        self._update_screen_state(self.screen_state)
        return self._get_reward(), False  # reward, terminated

    def _get_reward(self):
        reward = self.player.score - self.last_score
        self.last_score = self.player.score
        return reward

    def state_shape(self):
        return *self.screen_size, len(self.channels)

    def state(self):
        return self.screen_state

    def trigger_screen_state_redraw(self):
        self.screen_state = self._create_screen_state()

    def _create_screen_state(self):
        screen_state = np.zeros(self.state_shape(), dtype=bool)
        if self.game_state == GameState.in_maze:
            # Initialize status bar.
            screen_state[0, :, Env.channels['gauge_background']] = True
            # Let the maze initialize the rest.
            self.maze.initialize_screen_state(screen_state, self.player.has_torch())
        elif self.game_state == GameState.in_treasure_room:
            self.treasure_room.initialize_screen_state(screen_state)
        else:
            assert False
        return screen_state

    def _update_screen_state(self, state):
        state[:, :, Env.channels['player']] = False
        player_cell = Env.position_to_cell(self.player.player_pos)

        if self.game_state == GameState.in_maze:
            # Update status bar.
            width, height = self.screen_size
            state[0, :, Env.channels['gauge_health']] = False
            state[0, 0:self.player.health, Env.channels['gauge_health']] = True
            for gauge_channel in Env._inventory_item_to_gauge_channel.values():
                state[0, :, gauge_channel] = False
            for i, item in enumerate(self.player.inventory):
                state[0, width - i - 1, Env._inventory_item_to_gauge_channel[item]] = True

            state[player_cell[0] + 1, player_cell[1], Env.channels['player']] = True

            self.maze.update_screen_state(state, self.player.has_torch())
        elif self.game_state == GameState.in_treasure_room:
            state[player_cell[0], player_cell[1], Env.channels['player']] = True
            self.treasure_room.update_screen_state(state)
        else:
            assert False
        return state

    @staticmethod
    def position_to_cell(position):
        y, x = position
        return math.floor(y), math.floor(x)

    @staticmethod
    def cell_to_position(cell):
        return np.array([cell[0] + 0.5, cell[1] + 0.5], dtype=np.float32)

    def _get_checkpoint(self):
        return self.maze.get_checkpoint(self.player)

    def _apply_checkpoint(self, checkpoint):
        self.maze.apply_checkpoint(checkpoint, self.player)
        self.trigger_screen_state_redraw()

    def handle_human_action(self, action):
        action = action.lower()
        if action == 's':  # save checkpoint
            self.human_checkpoint = self._get_checkpoint()
        elif action == 'l':  # load checkpoint
            if self.human_checkpoint is not None:
                self._apply_checkpoint(self.human_checkpoint)


class GameState(Enum):
    in_maze = 0
    in_treasure_room = 1


_inventory_item_to_channel = {
    InventoryItem.key: Env.channels['key'],
    InventoryItem.torch: Env.channels['torch'],
    InventoryItem.amulet: Env.channels['amulet'],
    InventoryItem.sword: Env.channels['sword'],
}


class Maze:
    def __init__(self, environment):
        self.environment = environment
        # For documentation purposes, overridden in reset().
        self.rooms, self.room, self.soft_reset_cell, = \
            None, None, None
        self.pending_event = None
        self.reset()

    def reset(self):
        self.rooms = RoomCache()
        self._change_room(Env.initial_room)
        assert self.room.player_start is not None, 'Initial room does not specify initial player position.'
        self.soft_reset_cell = self.room.player_start

    def initialize_screen_state(self, screen_state, has_torch):
        for y in range(Room.room_size[1]):
            for x in range(Room.room_size[0]):
                if self.room.is_dark and not has_torch:
                    channel = 'darkness'
                else:
                    tile = self.room.room_data[y][x]
                    if tile not in _tile_to_channel:
                        continue
                    channel = _tile_to_channel[tile]
                screen_state[y + 1, x, Env.channels[channel]] = True

    def update_screen_state(self, screen_state, has_torch):
        self.room.update_screen_state(screen_state, has_torch)

    def update(self, player, action):
        player.update_inside_maze(action, self)
        self.room.update(player)
        if player.dead or self._has_collided(player):
            return MazeEvent.player_died
        if self.pending_event is not None:
            event = self.pending_event
            self.pending_event = None
            return event
        return None

    def _has_collided(self, player):
        player_cell = Env.position_to_cell(player.player_pos)
        if self.room.at(player_cell) == RoomTile.lava:
            return True
        return False

    def try_changing_room(self, player, new_player_cell):
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
            next_neighbour_name = self.room.neighbours[next_neighbour]
            if next_neighbour_name == 'treasure_room':
                self.pending_event = MazeEvent.entered_treasure_room
            else:
                self._change_room(next_neighbour_name)
                self.soft_reset_cell = next_location
                self.pending_event = MazeEvent.changed_room
            player.player_pos = Env.cell_to_position(next_location)
            player.player_speed = np.array([0, 0], dtype=np.float32)
            return True
        return False

    def _change_room(self, lvl_name):
        self.room = self.rooms.get_room(lvl_name)

    # TODO: PlayerCheckpoint, remove the argument
    def get_checkpoint(self, player):
        return MazeCheckpoint(
            room_name=self.room.room_name,
            soft_reset_cell=self.soft_reset_cell,
            player_pos=player.player_pos,
            player_speed=player.player_speed,
            player_state=player.player_state,
            exiting_ladder=player.exiting_ladder
        )

    # TODO: PlayerCheckpoint, remove the argument
    def apply_checkpoint(self, checkpoint, player):
        self._change_room(checkpoint.room_name)
        self.soft_reset_cell = checkpoint.soft_reset_cell
        player.player_pos = checkpoint.player_pos
        player.player_speed = checkpoint.player_speed
        player.player_state = checkpoint.player_state
        player.exiting_ladder = checkpoint.exiting_ladder


MazeCheckpoint = namedtuple('MazeCheckpoint',
                            ['room_name', 'soft_reset_cell', 'player_pos', 'player_speed',
                             'player_state', 'exiting_ladder'])


class MazeEvent(Enum):
    player_died = 0
    changed_room = 1
    entered_treasure_room = 2


class TreasureRoom:
    ticks_before_ending = 100

    def __init__(self, random):
        self.random = random
        self.ticks_since_start = 0
        self.coin_cell = None
        self._randomize_coin_position()

    # Returns whether the player's time in the treasure room has expired.
    def update(self, player, action):
        self.ticks_since_start += 1
        if self.ticks_since_start == TreasureRoom.ticks_before_ending:
            return True
        player.update_inside_treasure_room(action)
        player_cell = player.get_player_cell()
        if player_cell == self.coin_cell:
            player.score += Env.score_per_coin
            while player_cell == self.coin_cell:
                self._randomize_coin_position()

        return False

    def _randomize_coin_position(self):
        self.coin_cell = (self.random.randint(0, Room.room_size[1] - 1),
                          self.random.randint(0, Room.room_size[0] - 1))

    def initialize_screen_state(self, screen_state):
        screen_state[0, :, Env.channels['treasure_room_background']] = True

    def update_screen_state(self, screen_state):
        screen_state[:, :, Env.channels['coin']] = False
        screen_state[self.coin_cell[0], self.coin_cell[1], Env.channels['coin']] = True


class Room:
    room_size = (20, 19)  # (width, height)
    neighbours_names = ['left_neighbour', 'top_neighbour', 'right_neighbour', 'bottom_neighbour']

    def __init__(self, room_name, room_data, neighbours, is_dark):
        self.room_name = room_name
        self.room_data = room_data
        self.neighbours = neighbours
        self.is_dark = is_dark
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
        keys = [CollectableItem(cell, self, InventoryItem.key) for cell in keys_positions]
        amulets_positions = [(y, x)
                             for y in range(Room.room_size[1])
                             for x in range(Room.room_size[0])
                             if room_data[y][x] == RoomTile.amulet]
        amulets = [CollectableItem(cell, self, InventoryItem.amulet) for cell in amulets_positions]
        swords_positions = [(y, x)
                            for y in range(Room.room_size[1])
                            for x in range(Room.room_size[0])
                            if room_data[y][x] == RoomTile.sword]
        swords = [CollectableItem(cell, self, InventoryItem.sword) for cell in swords_positions]
        torches_positions = [(y, x)
                             for y in range(Room.room_size[1])
                             for x in range(Room.room_size[0])
                             if room_data[y][x] == RoomTile.torch]
        torches = [CollectableItem(cell, self, InventoryItem.torch) for cell in torches_positions]

        coins_positions = [(y, x)
                           for y in range(Room.room_size[1])
                           for x in range(Room.room_size[0])
                           if room_data[y][x] == RoomTile.coin]
        coins = [Coin(cell, self) for cell in coins_positions]

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

        self.moving_objects = enemies + doors + laser_doors + disappearing_walls
        self.scriptable_objects = self.moving_objects + keys + coins + amulets + swords + torches

        # Just for documentation, overridden in _update_moving_state.
        self.moving_data = None
        self._update_moving_state()

    def update(self, player):
        for o in self.scriptable_objects:
            o.update(player)
        self._update_moving_state()

    def _update_moving_state(self):
        moving_data = [[MovingObject.none
                        for _ in range(Room.room_size[0])]
                       for _ in range(Room.room_size[1])]
        for o in self.moving_objects:
            o.draw(moving_data)
        self.moving_data = moving_data

    def update_screen_state(self, state, has_torch):
        Enemy.reset_state(state)
        Door.reset_state(state)
        CollectableItem.reset_state(state)
        LaserDoor.reset_state(state)
        DisappearingWall.reset_state(state)
        Coin.reset_state(state)

        for o in self.scriptable_objects:
            if not self.is_dark or has_torch or o.visible_in_dark:
                o.add_to_state(state)

    def reset(self):
        for o in self.scriptable_objects:
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
        self.visible_in_dark = False

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
        if self.open or InventoryItem.key not in player.inventory:
            return
        top_y, top_x = self.top_position
        player_y, player_x = player.get_player_cell()
        if abs(player_x - top_x) <= 1 and top_y <= player_y < top_y + self.height:
            self.open = True
            player.inventory.remove(InventoryItem.key)


class CollectableItem:
    def __init__(self, position, room, item_type):
        self.position = position
        self.collected = False
        self.item_type = item_type
        self.visible_in_dark = False

    @staticmethod
    def reset_state(state):
        for channel in _inventory_item_to_channel.values():
            state[:, :, channel] = False

    def add_to_state(self, state):
        if self.collected:
            return
        y, x = self.position
        state[y + 1, x, _inventory_item_to_channel[self.item_type]] = True

    def update(self, player):
        if self.collected or player.is_inventory_full():
            return
        if player.get_player_cell() == self.position:
            self.collected = True
            player.collect(self.item_type)


# TODO: unite with CollectableItem
class Coin:
    def __init__(self, position, room):
        self.position = position
        self.collected = False
        self.visible_in_dark = False

    @staticmethod
    def reset_state(state):
        state[:, :, Env.channels['coin']] = False

    def add_to_state(self, state):
        if self.collected:
            return
        y, x = self.position
        state[y + 1, x, Env.channels['coin']] = True

    def update(self, player):
        if self.collected:
            return
        if player.get_player_cell() == self.position:
            self.collected = True
            player.score += 1000


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
        self.visible_in_dark = False

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
        self.visible_in_dark = True

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


class PlayerState(Enum):
    standing = 0
    flying = 1
    on_ladder = 2
    above_ladder = 3


class Player:
    _max_hearths = 5
    _inventory_size = 5

    def __init__(self):
        # For documentation purposes, overridden in reset() (which is called by the environment).
        self.player_speed, self.player_pos, self.player_state, self.exiting_ladder, self.health, self.dead, \
            self.inventory, self.score, self.amulet_active, self.ticks_since_amulet_activated = \
            None, None, None, None, None, None, None, None, None, None
        self._ignored_until_released = []

    def reset(self, position):
        self.soft_reset(position)
        self.health = Player._max_hearths
        #self.inventory = []
        self.inventory = [InventoryItem.key] * 3  # TODO
        self.score = 0
        self.amulet_active = False
        self.ticks_since_amulet_activated = 0  # only meaningful if self.amulet_active == True

    # called after player loses one hearth
    def soft_reset(self, position):
        self.player_pos = position
        self.player_speed = np.array([0, 0], dtype=np.float32)
        self.player_state = PlayerState.standing
        self.exiting_ladder = False
        self.dead = False

    def is_inventory_full(self):
        return len(self.inventory) == Player._inventory_size

    def has_torch(self):
        return InventoryItem.torch in self.inventory

    def collect(self, item):
        if item == InventoryItem.amulet:
            self.amulet_active = True
            self.ticks_since_amulet_activated = 0
        else:
            self.inventory.append(item)

    def _amulet_update(self):
        if self.amulet_active:
            self.ticks_since_amulet_activated += 1
            if self.ticks_since_amulet_activated == Env.amulet_duration:
                self.amulet_active = False
                self.ticks_since_amulet_activated = 0

    def get_player_cell(self):
        return Env.position_to_cell(self.player_pos)

    def die(self):
        self.dead = True

    def update_inside_treasure_room(self, action):
        new_player_pos = self.player_pos
        if action == 'left':
            new_player_pos[1] -= Env.treasure_room_walk_speed
        if action == 'right':
            new_player_pos[1] += Env.treasure_room_walk_speed
        if action == 'up':
            new_player_pos[0] -= Env.treasure_room_walk_speed
        if action == 'down':
            new_player_pos[0] += Env.treasure_room_walk_speed
        new_player_pos[0] = np.clip(new_player_pos[0], 0, Room.room_size[1] - 1)
        new_player_pos[1] = np.clip(new_player_pos[1], 0, Room.room_size[0] - 1)
        self.player_pos = new_player_pos

    def update_inside_maze(self, action, maze):
        self._amulet_update()
        if action in self._ignored_until_released:
            self._ignored_until_released = [action]
            action = 'nop'
        else:
            self._ignored_until_released = []

        print(self.player_state, self.player_pos, self.player_speed, action)
        curr_y, curr_x = self.get_player_cell()
        self._update_player_speed(maze, action)
        new_player_pos = self._calculate_new_position(maze, action)
        new_player_cell = Env.position_to_cell(new_player_pos)
        new_y, new_x = new_player_cell

        dy, dx = new_y - curr_y, new_x - curr_x
        total_delta = abs(dy) + abs(dx)
        y, x = curr_y, curr_x
        crashed = False
        for delta in range(1, total_delta):
            if dx != 0 and (x - curr_x) / dx < delta / total_delta:
                x += np.sign(new_x - curr_x)
            else:
                y += np.sign(new_y - curr_y)

            if not self._try_transitioning_to(np.array([y, x], dtype=np.float32), maze, action):
                crashed = True
                break
        if not crashed:
            self._try_transitioning_to(new_player_pos, maze, action)
        self.player_state = self._get_new_player_state(maze, action)

    def _try_transitioning_to(self, new_player_pos, maze, action):
        curr_room = maze.room
        curr_player_cell = self.get_player_cell()
        curr_y, curr_x = curr_player_cell
        new_player_cell = Env.position_to_cell(new_player_pos)
        new_y, new_x = new_player_cell
        crashed = False
        changed_room = False
        if not curr_room.is_inside(new_player_cell):
            changed_room = maze.try_changing_room(self, new_player_cell)
            crashed = not changed_room
        elif not (new_x == curr_x and new_y < curr_y) and \
                not curr_room.is_solid_at(curr_player_cell) and \
                curr_room.is_solid_at(new_player_cell):
            crashed = True
        elif self.player_state == PlayerState.above_ladder and \
                (new_x == curr_x and new_y > curr_y) and \
                action != 'down':
            crashed = True

        if crashed:
            print(self.player_speed[0], end='')
            if self.player_speed[0] > 1:
                print(' -> died')
            else:
                print()

        transitioned = not crashed and not changed_room
        if transitioned:
            self.player_pos = new_player_pos
        is_on_ladder = curr_room.at(self.get_player_cell()) == RoomTile.ladder and not self.exiting_ladder
        if not transitioned or is_on_ladder:
            self.player_speed[0] = 0
            return False
        return True

    def _update_player_speed(self, maze, action):
        standing, on_ladder, flying, above_ladder = self._one_hot_state()
        if standing or on_ladder or above_ladder:
            if action == 'jump':
                # Position the player to the center of current tile, to make sure every jump occurs from the same y-position.
                self.player_pos[0] = math.floor(self.player_pos[0]) + 0.5
                self.player_speed[0] -= Env.jump_force
                self.player_state = PlayerState.flying
                if maze.room.at(self.get_player_cell()) == RoomTile.ladder:
                    self.exiting_ladder = True
        elif flying:
            self.player_speed[0] += Env.gravity

    # Calculates new position of the player not counting in collisions.
    def _calculate_new_position(self, maze, action):
        # Player's position has floating-point coordinates that get floored when displaying. Physics behaves as if the
        # player was a single point.
        standing, on_ladder, flying, above_ladder = self._one_hot_state()
        player_cell = self.get_player_cell()
        cell_bellow = (player_cell[0] + 1, player_cell[1])
        curr_room = maze.room

        new_player_pos = self.player_pos + self.player_speed
        if standing or on_ladder or above_ladder:
            if action == 'left':
                new_player_pos[1] -= Env.walking_speed
            elif action == 'right':
                new_player_pos[1] += Env.walking_speed

        if flying:
            if action == 'left':
                new_player_pos[1] -= Env.lateral_jumping_speed
            elif action == 'right':
                new_player_pos[1] += Env.lateral_jumping_speed

        if on_ladder:
            if action in ['jump', 'left', 'right']:
                self.exiting_ladder = True
            else:
                if action == 'up':
                    new_player_pos[0] -= Env.ladder_speed
                elif action == 'down':
                    new_player_pos[0] += Env.ladder_speed

        if above_ladder and action == 'down':
            new_player_pos[0] += Env.ladder_speed

        if standing:
            if curr_room.at(cell_bellow) == RoomTile.moving_sand:
                new_player_pos[1] -= Env.moving_sand_speed

        return new_player_pos

    def _get_new_player_state(self, maze, action):
        curr_room = maze.room
        player_cell = self.get_player_cell()
        cell_bellow = (player_cell[0] + 1, player_cell[1])

        can_be_on_ladder = curr_room.at(player_cell) == RoomTile.ladder
        flying_up = self.player_speed[0] < -10e-3
        can_stand_on_ladder = not flying_up and curr_room.at(cell_bellow) == RoomTile.ladder and not can_be_on_ladder
        can_stand = not flying_up and (curr_room.at(cell_bellow) in [RoomTile.wall, RoomTile.moving_sand] or
                                       can_stand_on_ladder or
                                       curr_room.at_moving(cell_bellow) in [MovingObject.disappearing_wall,
                                                                            MovingObject.door])

        if self.player_state == PlayerState.flying or (self.player_state == PlayerState.standing and action == 'up'):
            if can_be_on_ladder and not self.exiting_ladder:
                self._ignore_until_released('left', 'right', 'jump')
                return PlayerState.on_ladder
        if self.player_state == PlayerState.flying:
            if can_stand_on_ladder:
                return PlayerState.above_ladder
            if can_stand:
                return PlayerState.standing
        if self.player_state == PlayerState.standing:
            if can_stand_on_ladder:
                return PlayerState.above_ladder
            if not can_stand:
                return PlayerState.flying
        if self.player_state == PlayerState.above_ladder:
            if not can_stand_on_ladder and can_stand:
                return PlayerState.standing
            if can_be_on_ladder:
                return PlayerState.on_ladder
            if not can_stand:
                return PlayerState.flying
        if self.player_state == PlayerState.on_ladder:
            # Allow transitioning to the level below by climbing a ladder.
            if can_stand and curr_room.is_inside(cell_bellow):
                return PlayerState.standing
            if not can_be_on_ladder and can_stand_on_ladder:
                self.exiting_ladder = False
                return PlayerState.above_ladder
            if self.exiting_ladder or not can_be_on_ladder:
                self.exiting_ladder = False
                return PlayerState.standing if can_stand else PlayerState.flying

        if not can_be_on_ladder or can_stand or can_stand_on_ladder:
            self.exiting_ladder = False
        return self.player_state

    def _ignore_until_released(self, *actions):
        for action in actions:
            if action not in self._ignored_until_released:
                self._ignored_until_released.append(action)

    def _one_hot_state(self):
        return self.player_state == PlayerState.standing, self.player_state == PlayerState.on_ladder, \
               self.player_state == PlayerState.flying, self.player_state == PlayerState.above_ladder


class Enemy:
    _ticks_per_move = 2  # inverse of the speed of an enemy

    def __init__(self, starting_cell, room):
        self.starting_cell = starting_cell
        self.room = room
        # For documentation purposes, overridden in reset().
        self.enemy_cell, self.ticks_since_move, self.previous_cell = None, None, None
        self.reset()
        self.visible_in_dark = True

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
        if player.get_player_cell() == self.enemy_cell and not player.amulet_active:
            player.die()

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
            is_dark = False if 'is_dark' not in data else data['is_dark']
            return Room(room_name, room_data, neighbours, is_dark)

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
    laser_door = 3
    disappearing_wall = 4


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
    amulet = 10
    sword = 11
    torch = 12
    laser_door = 13
    disappearing_wall = 14
    coin = 15


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
    _hex(0x4d0404): RoomTile.amulet,
    _hex(0x3c4d04): RoomTile.sword,
    _hex(0xaaaf97): RoomTile.torch,
    _hex(0x584052): RoomTile.laser_door,
    _hex(0x626262): RoomTile.disappearing_wall,
    _hex(0xff9400): RoomTile.coin
}
