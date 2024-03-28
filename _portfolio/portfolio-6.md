---
title: "Elixir/Phoenix App using Redis on Fly"
collection: portfolio
---

Date of writing: 30.8.2022

Recently, I worked on creating [a tutorial](https://upstash.com/docs/redis/quickstarts/elixir) demonstrating how to deploy an Elixir/Phoenix web app that uses Redis on [Fly](https://fly.io/).

This project was particularly exciting for me because I had the chance to delve into Elixir, a functional programming language. It marked my first venture into building something with functional languages outside of my university coursework.

For communication with Upstash Redis, I utilized [Redix](https://github.com/whatyouhide/redix), an Elixir-based Redis client. While originally designed for Redis, it seamlessly works with Upstash Redis since the Upstash Redis API adheres to the Redis API.

Ultimately, I created [a simple app that provides weather information upon user request](https://elixir-redis.fly.dev/). The app uses Redis to cache responses, which offers several advantages. By reducing the number of calls to the external API (in this case, [WeatherAPI](https://www.weatherapi.com/)), Redis helps to trim response times and cut down on costs.
