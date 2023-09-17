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
    a & = \frac{1}{4} + \frac{1}{4} + \frac{b}{4} + \frac{d}{4} \\
    b & = \frac{1}{4} + \frac{a}{4} + \frac{c}{4} + \frac{E}{4} \\
\end{align}

$$

For $a$, we have two $\frac{1}{4}$ terms because ant will reach a border if it moves left or up when it is on position $a$. Otherwise it will go to $b$ with probability $\frac{1}{4}$, hence the average time $\frac{b}{4}$. Same is the case for $d$. With this approach, average time to reach a border from $E$ will be:

$$

\begin{align}
    E & = \frac{b}{4} + \frac{d}{4} + \frac{f}{4} + \frac{h}{4} \\
\end{align}

$$

Only thing left to do, is to figure out how to calculate all these variables. Here, we will **reduce our problem to a linear algebra problem**.

### Implementation

### Test