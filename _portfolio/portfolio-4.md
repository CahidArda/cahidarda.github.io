---
title: "Optiver Quantitative Researcher Sample Questions"
excerpt: "My solutions to the sample questions provided by Optiver for their Quantitive Researcher position"
collection: portfolio
---

Date of writing: 17.9.2023

In this post, I will share my solutions to the sample questions Optiver has shared for their Quantitive Researcher position. I am assuming that sharing my solutions is not a problem because answers of these questions are not likely to be part of the interview process and there is no statement asking to not share solutions in the job listing. If requested, I can remove my solutions from my webpage.

# Introduction

Optiver provides the following questions in their [job listing](https://optiver.com/working-at-optiver/career-opportunities/6904662002/):

> An ant leaves its anthill in order to forage for food. It moves with the speed of 10cm per second, but it doesn’t know where to go, therefore every second it moves randomly 10cm directly north, south, east or west with equal probability.
>
> - If the food is located on east-west lines 20cm to the north and 20cm to the south, as well as on north-south lines 20cm to the east and 20cm to the west from the anthill, how long will it take the ant to reach it on average?
> - What is the average time the ant will reach food if it is located only on a diagonal line passing through (10cm, 0cm) and (0cm, 10cm) points?
> - Can you write a program that comes up with an estimate of average time to find food for any closed boundary around the anthill? What would be the answer if food is located outside an area defined by ( (x – 2.5cm) / 30cm )2 + ( (y – 2.5cm) / 40cm )2 < 1 in coordinate system where the anthill is located at (x = 0cm, y = 0cm)?

In each question, I will have three sections:
1. Solution: This is where I explain my solution
2. Implementation: This is where I write the script which will solve this problem for me with my solution
3. Test: This is where I will run experiments to verify the correctness of my solution

## Question 1

In question 1, our ant is in the middle of a box of shape 40x40 cm. We need to calculate the average time it will take our ant to reach any border of this box.

### Solution

Since our ant moves 10 cm per second in any direction, I will divide our box into a 4x4 grid. When we exclude the points which are on the border, there will be 9 points on the grid. I will name these points with variables:

$$

\begin{bmatrix}
    a & b & c \\
    d & E & f \\
    g & h & i \\
\end{bmatrix}

$$

$E$ is written in capital because it is where our ant is originally. Let each of these letters denote the *average time it will take the ant to move from that position to reach the border*. Then we can write the average time for $a$ and $b$ like the following:

$$

\begin{align}
    a & = \frac{1}{4} + \frac{1}{4} + \frac{1+b}{4} + \frac{1+d}{4} \\
    b & = \frac{1}{4} + \frac{1+a}{4} + \frac{1+c}{4} + \frac{1+E}{4} \\
\end{align}

$$

For $a$, we have two $\frac{1}{4}$ terms because ant will reach a border if it moves left or up when it is on position $a$. Otherwise it will go to $b$ with probability $\frac{1}{4}$, hence the average time $\frac{1}{4} * (1+b)$. Same is the case for $d$. With this approach, average time to reach a border from $E$ will be:

$$

\begin{align}
    E & = \frac{1+b}{4} + \frac{1+d}{4} + \frac{1+f}{4} + \frac{1+h}{4} \\
\end{align}

$$

Only thing left to do, is to figure out how to calculate all these variables. Here, we will **reduce our problem to a linear algebra problem**. We can write the equation above for every variable on our grid. Then, we can collect the constants on the right side and variables on the left side of the equation. Denote the left side of the equation with matrix $\underset{3\times 3}{\mathrm{A}}$ and right side with matrix $\underset{1\times 3}{\mathrm{b}}$. $\mathrm{A}$ matrix looks like this:

$$

\mathrm{A} = \begin{bmatrix}
   1.   & -0.25 &  0.   & -0.25 &  0.   &  0.   &  0.   &  0.   &  0.   \\
  -0.25 &  1.   & -0.25 &  0.   & -0.25 &  0.   &  0.   &  0.   &  0.   \\
   0.   & -0.25 &  1.   &  0.   &  0.   & -0.25 &  0.   &  0.   &  0.   \\
  -0.25 &  0.   &  0.   &  1.   & -0.25 &  0.   & -0.25 &  0.   &  0.   \\
   0.   & -0.25 &  0.   & -0.25 &  1.   & -0.25 &  0.   & -0.25 &  0.   \\
   0.   &  0.   & -0.25 &  0.   & -0.25 &  1.   &  0.   &  0.   & -0.25 \\
   0.   &  0.   &  0.   & -0.25 &  0.   &  0.   &  1.   & -0.25 &  0.   \\
   0.   &  0.   &  0.   &  0.   & -0.25 &  0.   & -0.25 &  1.   & -0.25 \\
   0.   &  0.   &  0.   &  0.   &  0.   & -0.25 &  0.   & -0.25 &  1.   \\
\end{bmatrix}

$$

$\mathrm{b}$ is simply:

$$

\mathrm{b} = \begin{bmatrix}
    1. \\
    1. \\
    1. \\
    1. \\
    1. \\
    1. \\
    1. \\
    1. \\
    1. \\
\end{bmatrix}

$$

Now that we have a solution proposal, we can go ahead with implementation of this solution.

### Implementation

In the implementation, I didn't want to write out the $\mathrm{A}$ and $\mathrm{b}$ matrices by hand to avoid any mistakes.

I began my solution by defining some variables. `box_size` denotes the size of the box our ant is in. `grid_size` denotes the size of the variable grid.

```python
box_size = 4
grid_size = box_size - 1
variable_count = grid_size * grid_size
```

Then I define functions which will make it easier to initialize our matrices

```python
def adjacent_border_count(x: int, y: int) -> int:
    """
    Given a point on the grid, returns the number of border
    points adjacent to it
    """
    return (
        (x==0 or x==(grid_size-1))
        + (y==0 or y==(grid_size-1))
    )


def point_to_index(x: int, y: int) -> int:
    """
    Given a point on the grid, returns the index of the cell for the
    matrixes.
    """
    return y * grid_size + x


def adjacent_var_indexes(x: int, y: int) -> Iterator[int]:
    """
    Given a point on the grid, returns the indexes of the adjacent cells.
    """
    on_grid = lambda p: ((p >= 0) and (p < grid_size))

    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        _x, _y = x + dx, y + dy

        if on_grid(_x) and on_grid(_y):
            yield point_to_index(_x, _y)
```

Finally, I initialize the matrices and calculate the solution


```python
# init matrices
A = np.zeros((variable_count, variable_count))
b = np.zeros((variable_count))


# fill matrices
for y in range(grid_size):
    for x in range(grid_size):
        _index = point_to_index(x, y)

        # update A
        A[_index, _index] = 1
        print("  ", x, y, _index)
        for adjacent_index in adjacent_var_indexes(x, y):
            print(adjacent_index)
            A[adjacent_index, _index] = -1/4
        print()
        # update b
        b[_index] = 1 # 1/4 * adjacent_border_count(x, y)

print(A)
print(b)
print(np.linalg.solve(A, b))
```

Output of my implementation is as follows:

```
[[ 1.   -0.25  0.   -0.25  0.    0.    0.    0.    0.  ]
 [-0.25  1.   -0.25  0.   -0.25  0.    0.    0.    0.  ]
 [ 0.   -0.25  1.    0.    0.   -0.25  0.    0.    0.  ]
 [-0.25  0.    0.    1.   -0.25  0.   -0.25  0.    0.  ]
 [ 0.   -0.25  0.   -0.25  1.   -0.25  0.   -0.25  0.  ]
 [ 0.    0.   -0.25  0.   -0.25  1.    0.    0.   -0.25]
 [ 0.    0.    0.   -0.25  0.    0.    1.   -0.25  0.  ]
 [ 0.    0.    0.    0.   -0.25  0.   -0.25  1.   -0.25]
 [ 0.    0.    0.    0.    0.   -0.25  0.   -0.25  1.  ]]

[1. 1. 1. 1. 1. 1. 1. 1. 1.]

[2.75 3.5  2.75 3.5  4.5  3.5  2.75 3.5  2.75]
```

The middle element of our solution is **4.5**, which denoted the average time it will take our ant to move from the middle of the box to any border. Making our solution, 4.5.

<details>
<summary>Complete Code</summary>

```python
import numpy as np
from typing import Iterator


# variables
box_size = 4
grid_size = box_size - 1
variable_count = grid_size * grid_size


def adjacent_border_count(x: int, y: int) -> int:
    """
    Given a point on the grid, returns the number of border
    points adjacent to it
    """
    return (
        (x==0 or x==(grid_size-1))
        + (y==0 or y==(grid_size-1))
    )


def point_to_index(x: int, y: int) -> int:
    """
    Given a point on the grid, returns the index of the cell for the
    matrixes.
    """
    return y * grid_size + x


def adjacent_var_indexes(x: int, y: int) -> Iterator[int]:
    """
    Given a point on the grid, returns the indexes of the adjacent cells.
    """
    on_grid = lambda p: ((p >= 0) and (p < grid_size))

    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        _x, _y = x + dx, y + dy

        if on_grid(_x) and on_grid(_y):
            yield point_to_index(_x, _y)
            

# init matrices
A = np.zeros((variable_count, variable_count))
b = np.zeros((variable_count))


# fill matrices
for y in range(grid_size):
    for x in range(grid_size):
        _index = point_to_index(x, y)

        # update A
        A[_index, _index] = 1
        print("  ", x, y, _index)
        for adjacent_index in adjacent_var_indexes(x, y):
            print(adjacent_index)
            A[adjacent_index, _index] = -1/4
        print()
        # update b
        b[_index] = 1 # 1/4 * adjacent_border_count(x, y)


print(A)
print(b)
print(np.linalg.solve(A, b))
```
</details>


### Test

Before concluding my work on question 1, I would like to run an experiment to check if it really takes 4.5 seconds on average for the ant to reach the border of our box. To do this, I will simply simulate our ant and run several simulations and take the average.

To test my results, I have written the following script:

```python
from random import randint
from typing import Tuple

# variables
box_size = 4
number_of_simulations = [1, 10, 100, 1000, 10000, 100000]
border_distance = box_size//2

assert (box_size%2)==0, "Box size should be an even number"


def get_random_move() -> Tuple[int, int]:
    """
    Generates a random move which can be one of the following:
    (0, 1), (1, 0), (0, -1), (-1, 0)
    """
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    return moves[randint(0, 3)]


def on_border(x: int, y: int) -> bool:
    """
    Checks if ant is border_distance away from starting position
    (0,0)
    """
    return any([
        abs(x) == border_distance,
        abs(y) == border_distance
    ])


def run_simulation() -> int:
    """
    Runs a simulation and returns the number of seconds it took
    for the and the reach a border

    Ant starts at position (0,0) and simulation ends when ant reaches
    a border.
    """
    position = [0, 0]
    time = 0
    while not on_border(*position):
        move = get_random_move()
        position = [p+dp for p, dp in zip(position, move)]
        time += 1
    return time


def run_simulations(sim_count: int) -> Tuple[int, int]:
    """
    Runs sim_count many simulations to calculate
    average and maximum time for the ant to reach a border.
    """
    sim_times = [
        run_simulation() for i in range(sim_count)
    ]

    return (
        sum(sim_times) / sim_count,
        max(sim_times)
    )


# result
print("sim_count     mean      max")
for sim_count in number_of_simulations:
    mean_time, max_time = run_simulations(sim_count)
    print(
        str(sim_count).rjust(9),
        str(mean_time).rjust(8),
        str(max_time).rjust(8))
```

The output is as follows:

```
sim_count     mean      max
        1      3.0        3
       10      5.5       11
      100     4.42       16
     1000    4.437       27
    10000   4.5189       33
   100000  4.50917       38
```

Looking at the results of my test, I conclude that my solution and implementation are correct.

## Question 2

Not done yet

## Question 3

Not done yet
