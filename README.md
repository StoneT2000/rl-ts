# Typescript Reinforcement Learning Library 🤖

[![npm version](https://badge.fury.io/js/rl-ts.svg)](https://badge.fury.io/js/rl-ts)

This is a RL library built with **typescript**, enabling faster bug-free development and scalability and more.

Install with 

```
npm install rl-ts
```

Web package that you can import using `<script></script>` coming soon!

To get started - WIP

## Features

- Uses a **Open AI Gym** like interface to provide an accessible and general approach for developing environments and testing models

- Provides standard environments

- Provides a toolbox of agents

- Did I mention that it's written in **typescript**? That means types! Less bugs!

Why RL with typescript / javascript (TS/JS)? TS/JS powers the web and a ton of the awesome visuals you see online. 
While indeed production level reinforcement deep learning should be done with python / C++, TS/JS makes it easy to directly integrate RL 
into the browser and serve not only as a simple integration for visualizers, but also as a powerful web-based teaching tool on RL. Moreover, this library can help developers with a background in TS/JS ease their way into using python / C++ for RL.

Inspired by [Andrej Karpath's blog post](http://karpathy.github.io/2016/05/31/rl/), I'm building this to get a strong, end to end understanding of deep learning and reinforcement learning. While one could just use his library or some of the other ones out there, none of them are built with typescript / actively maintained. Typescript enables typing which massively improves the scalability and maintainability of this library.

## Road map and plans

This library aims to be effectively a combination of the Open AI Gym and a miniature Stable Baselines, with more emphasis on naive RL algorithms to be used as a educational resource and help build more powerful visualizers that are much more accessible.

## Development

First install all necessary dependencies via 

```
npm i
```

To build the library run

```
npm run build
```

To run tests run

```
npm run tests
```