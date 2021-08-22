# MinAtar
MinAtar is a testbed for AI agents which implements miniaturized versions of several Atari 2600 games. MinAtar is inspired by the Arcade Learning Environment (Bellemare et. al. 2013) but simplifies the games to make experimentation with the environments more accessible and efficient. Currently, MinAtar provides analogues to five Atari games which play out on a 10x10 grid. The environments provide a 10x10xn state representation, where each of the n channels correspond to a game-specific object, such as ball, paddle and brick in the game Breakout.

<p  align="center">
<img src="img/seaquest.gif" width="200" />
<img src="img/breakout.gif" width="200" />
</p>
<p  align="center">
<img src="img/asterix.gif" width="200" />
<img src="img/freeway.gif" width="200" />
<img src="img/space_invaders.gif" width="200" />
</p>

## Quick Start
To use MinAtar, you need python3 installed, make sure pip is also up to date.  To run the included `DQN` and `AC_lambda` examples, you need `PyTorch`.  To install MinAtar, please follow the steps below:

1. Clone the repo: 
```bash
git clone https://github.com/kenjyoung/MinAtar.git
```
If you prefer running MinAtar in a virtualenv, you can do the following before step 2:
```bash
python3 -m venv venv
source venv/bin/activate
# Upgrade Pip
pip install --upgrade pip
```

2.  Install MinAtar:
```bash
pip install .
```
If you have any issues with automatic dependency installation, you can instead install the necessary dependencies manually and run
```bash
pip install . --no-deps
```

To verify the installation is successful, run
```bash
python examples/random_play.py -g breakout
```
The program will run 1000 episodes with a random policy and report the mean and standard error in the resulting returns similar to:
```bash
Avg Return: 0.5+/-0.023194827009486406
```

The examples/random_play.py is a simple example to demonstrate how to use the module. `breakout` in the previous command can be replaced by one of the five available games: asterix, breakout, freeway, seaquest and space_invaders. See the Games section below for details of each game.

To play a game as a human, run examples/human_play.py as follows:

```bash
python human_play.py -g <game>
```
Use the arrow keys to move and space bar to fire. Also, press q to quit and r to reset.

Also included in the examples directory are example implementations of DQN (dqn.py) and online actor-critic with eligibility traces (AC_lambda.py).

## Visualizing the Environments
We provide 2 ways to visualize a MinAtar environment.
### Using Environment.display_state()
The Environment class includes a simple visualizer using matplotlib in the display_state function. To use this simply call:
```python
env.display_state(50)
```
where env is an instance of MinAtar.Environment. The argument is the number of milliseconds to display the state before continuing execution. To close the resulting display window call:
```python
env.close_display()
```
This is the simplest way to visualize the environments, unless you need to handle user input during execution in which case you could use the provided GUI class.


### Using GUI class
We also include a slightly more complex GUI to visualize the environments and optionally handle user input. This GUI is used in examples/human_play.py to play as a human and examples/agent_play.py to visualize the performance of trained agents. To use the GUI you can import it in your code with:
```python
from minatar import GUI
```
Initialize an instance of the GUI class by providing a name for the window, and the integer number of input channels for the minatar environment to be visualized. For example:
```python
GUI(env.game_name(), env.n_channels)

```
where env is an instance of minatar.Environment. The recommended way to use the GUI for visualizing an environment is to include all you're agent-environment interaction code in a function that looks something like this:
```python
def func():
    gui.display_state(env.state())
    #One step of agent-environment interaction here
    gui.update(50, func)
```
The first argument to gui.update is the time to hold the current frame before continuing. The second argument specifies the function to call after that time has elapsed. In the example above the call to update simply calls func again, effectively continuing the agent-environment interaction loop. Note that this is not a recursive call, as the call to func in update is made in a new thread, while the execution of the current thread continues.

To begin the execution you can use:
```python
gui.update(0, func)
gui.run()
```
This will enter the agent environment interaction loop and then run the GUI thread, gui.run() will block until gui.quit() is called. To handle user input you can use gui.overwrite_key_handle(on_key_event, on_release_event). The arguments are functions to be called whenever a key is pressed, and released respectively. For an example of how to do this see examples/human_play.py.

## Support for Other Languages

- [Julia](https://github.com/mkschleg/MinAtar.jl/blob/master/README.md)

## Results
The following plots display results for DQN (Mnih et al., 2015) and actor-critic (AC) with eligibility traces. Our DQN agent uses a significantly smaller network compared to that of Mnih et al., 2015. We display results for DQN with and without experience reply. Our AC agent uses a similar architecture to DQN, but does not use experience replay. We display results for two values of the trace decay parameter, 0.8 and 0.0.  Each curve is the average of 30 independent runs with different random seeds. The top plots display the sensitivity of final performance to the step-size parameter, while the bottom plots display the average return during training as a function of training frames. For further information, see the paper on MinAtar available [here](https://arxiv.org/abs/1903.03176).

<img align="center" src="img/sensitivity_curves.gif" width=800>
<img align="center" src="img/learning_curves.gif" width=800>

## Games
So far we have implemented analogues to five Atari games in MinAtar as follows. For each game, we include a link to a video of a trained DQN agent playing.

### Asterix
The player can move freely along the 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of +1 is given for picking up treasure. Termination occurs if the player makes contact with an enemy. Enemy and treasure direction are indicated by a trail channel. Difficulty is periodically increased by increasing the speed and spawn rate of enemies and treasure.

[Video](https://www.youtube.com/watch?v=Eg1XsLlxwRk)

### Breakout
The player controls a paddle on the bottom of the screen and must bounce a ball to break 3 rows of bricks along the top of the screen. A reward of +1 is given for each brick broken by the ball.  When all bricks are cleared another 3 rows are added. The ball travels only along diagonals. When the ball hits the paddle it is bounced either to the left or right depending on the side of the paddle hit. When the ball hits a wall or brick, it is reflected. Termination occurs when the ball hits the bottom of the screen. The ball's direction is indicated by a trail channel.

[Video](https://www.youtube.com/watch?v=cFk4efZNNVI&t)

### Freeway
The player begins at the bottom of the screen and the motion is restricted to travelling up and down. Player speed is also restricted such that the player can only move every 3 frames. A reward of +1 is given when the player reaches the top of the screen, at which point the player is returned to the bottom. Cars travel horizontally on the screen and teleport to the other side when the edge is reached. When hit by a car, the player is returned to the bottom of the screen. Car direction and speed is indicated by 5 trail channels.  The location of the trail gives direction while the specific channel indicates how frequently the car moves (from once every frame to once every 5 frames). Each time the player successfully reaches the top of the screen, the car speeds are randomized. Termination occurs after 2500 frames have elapsed.

[Video](https://www.youtube.com/watch?v=gbj4jiTcryw)

### Seaquest
The player controls a submarine consisting of two cells, front and back, to allow direction to be determined. The player can also fire bullets from the front of the submarine. Enemies consist of submarines and fish, distinguished by the fact that submarines shoot bullets and fish do not. A reward of +1 is given each time an enemy is struck by one of the player's bullets, at which point the enemy is also removed. There are also divers which the player can move onto to pick up, doing so increments a bar indicated by another channel along the bottom of the screen. The player also has a limited supply of oxygen indicated by another bar in another channel. Oxygen degrades over time and is replenished whenever the player moves to the top of the screen as long as the player has at least one rescued diver on board. The player can carry a maximum of 6 divers. When surfacing with less than 6, one diver is removed. When surfacing with 6, all divers are removed and a reward is given for each active cell in the oxygen bar. Each time the player surfaces the difficulty is increased by increasing the spawn rate and movement speed of enemies. Termination occurs when the player is hit by an enemy fish, sub or bullet; or when oxygen reaches 0; or when the player attempts to surface with no rescued divers. Enemy and diver directions are indicated by a trail channel active in their previous location to reduce partial observability.

[Video](https://www.youtube.com/watch?v=W9k38b5QPxA&t)

### Space Invaders
The player controls a cannon at the bottom of the screen and can shoot bullets upward at a cluster of aliens above. The aliens move across the screen until one of them hits the edge, at which point they all move down and switch directions. The current alien direction is indicated by 2 channels (one for left and one for right) one of which is active at the location of each alien. A reward of +1 is given each time an alien is shot, and that alien is also removed. The aliens will also shoot bullets back at the player. When few aliens are left, alien speed will begin to increase. When only one alien is left, it will move at one cell per frame. When a wave of aliens is fully cleared, a new one will spawn which moves at a slightly faster speed than the last. Termination occurs when an alien or bullet hits the player.

[Video](https://www.youtube.com/watch?v=W-9Ru-RDEoI)

### Montezuma's Revenge
The player navigates through a complex world with different kinds of enemies,
traps, door+key systems, and rewards. Only one section of the world -- a room
-- is displayed at any given time and the player can transition between the
rooms by walking to the edge of the screen. There are 23 regular rooms and 1
treasure room arranged in a pyramidal shape. See
`docs/montezumas_revenge-map.png`.
When navigating the regular rooms, the player can jump and move left or right.
They have an inventory, which can keep at most 5 items at a time.
The mechanism in those rooms are:
- Ropes/Ladders -- The player can jump on or climb the rope.
- Conveyor belts -- When the player stands on them, they push them to the left.
- Disappearing floors -- Periodically appearing and disappearing.
- Coins/Jewels -- Collecting them delivers a reward of +1000. They can't be
  collected when the inventory is full, although they don't occupy space in it
  once collected.
- Keys and doors -- Doors can only be opened if a key is in the inventory. Key
  can be collected by simply touching it.
- Laser doors - They periodically appear and disappear. Touching them results in
  death.
- Lava - Touching it results in death.
- Skulls - They either bounce or patrol.
- Snakes - They standstill.
- Spiders - They patrol. Touching a skull, a snake, or spider results in death
  of both the player and the creature.
- Amulets - After collecting an amulet, the player is immune to enemies for 50
  ticks.
- Swords - After collecting a sword, it appears in the inventory. After that, if
  the player touches a skull or a spider, the enemy is killed, the sword is
  discarded and a reward is allocated -- +2000 for a skull or +3000 for a
  spider. Snakes are immune to swords.
- Torches - After collecting a torch, it appears in the inventory. Some rooms in
  the maze are "dark", meaning that when the player enters, they can only see
  themselves, enemies, lava, and disappearing floors. However, if they possess a
  torch, they can see everything.

Each of those mechanisms is represented by a dedicated channel.
Besides touching lava, a laser door, or an enemy, the player dies if they fall from too high.
The game ends after the player loses 5 hearts. The current number of hearts and
the current content of the inventory are both displayed in the upper row of the
screen (hearts on the left and inventory on the right).

If the player manages to enter the treasure room (requiring them to collect all
keys and probably a torch), they get to be in it for 100 ticks before the game
is restarted. The physics mode changes - the player can move in any of the 4
directions freely and no gravity is present. 10 lava blocks and a jewel are at
random locations. The player can collect the jewel. After that, it moves to
another location at random. If the player touches the lava, they die.

To make the game easier to explore, human agents can press 's' and 'l' to create
(save) or load a checkpoint. After loading a checkpoint, the player is moved to the
saved location, but the state of the world is not restored (meaning doors,
enemies, etc.).

The layout and metadata of all rooms are described by `json` and `png` files in
`data/montezumas_revenge`. The color coding used is specified in
`docs/montezumas_revenge-color_coding.png`.  The `png` files must be PNG images,
20 x 19, 8-bit/color RGB, non-interlaced. Be sure not to include the alpha
channel (RGBA).

### Ms. Pac-Man
The player navigates through a maze, picking up coins along the way. Each coin
is worth +10 reward. The player can change their direction but can only stop
when they bump into a wall. There are 4 enemies in the maze moving randomly.
They spawn in the center of the maze. If the player collides with an enemy,
they lose one of 3 hearts, and positions of both the player and all enemies
reset. When all hearts are lost, the game ends.
There are 4 power pills positioned in the corners of the maze. If the player
collects one of them, (1) a reward of +50 is assigned and (2) for the next 50
ticks, colliding with an enemy kills the enemy (instead of the player) and the
player gets a reward. For killing the 1st, 2nd, 3rd, or 4th enemy, the player
receives a reward of +200, +400, +800, and +1600 respectively. 20 ticks after an
enemy is killed, it respawns in its original location.
The game is hard to play for a human agent since for efficiency reasons, each
square represents a piece of the maze instead of being either wall or empty.
More precisely, each square specifies in which directions an agent can depart
from it. These types of squares are represented by one channel each. Special
channels are dedicated to coins and power pills.
Number of hearts remaining is displayed in the left part of the first row of
the screen.
The shape of the maze is loaded from `data/pacman/level.json` and
`data/pacman/level.png`. The color coding used for specifying the shape can be
found in `docs/pacman-color_coding.png`. For humans, `docs/pacman-maze.png`
shows the shape of the maze in a friendlier way.

## Citing MinAtar
If you use MinAtar in your research please cite the following:

Young, K. Tian, T. (2019). MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments.  *arXiv preprint arXiv:1903.03176*.

In BibTeX format:

```
@Article{young19minatar,
author = {{Young}, Kenny and {Tian}, Tian},
title = {MinAtar: An Atari-Inspired Testbed for Thorough and Reproducible Reinforcement Learning Experiments},
journal = {arXiv preprint arXiv:1903.03176},
year = "2019"
}
```


## References
Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The arcade learning environment: An evaluation platform for general agents. *Journal of Artificial Intelligence Research*, 47, 253–279.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., . . . others (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529.

## License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
