#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module solves the maze."""

import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
from PIL import Image

COLORS = [(128, 0, 0), (0, 128, 128), (0, 0, 128), (230, 25, 75), (245, 130,
                                                                   48),
          (255, 255, 25), (210, 245, 60), (60, 180, 75), (70, 240, 240),
          (0, 130, 200), (145, 30, 180), (240, 50, 230), (250, 190, 190),
          (255, 214, 180), (170, 255, 195), (230, 190, 255)]


def load_maze_arr():

    def collapse(arr):
        stack = [arr[0]]
        for x in range(1, arr.shape[0]):
            if not all(stack[-1] == arr[x]):
                stack.append(arr[x])
        return np.stack(stack)

    im = Image.open('maze.png').convert('L')
    large = np.array(list(im.getdata())).reshape(im.size) // 255
    small = collapse(large)
    return collapse(small.transpose()).transpose()


def find_goals(maze_arr):
    goals = set()
    for x in range(2, maze_arr.shape[0] - 2):
        for y in range(2, maze_arr.shape[1] - 2):
            if np.sum(maze_arr[x - 2:x + 3, y - 2:y + 3]) == 25:
                goals.add((x, y))
    return goals


def convert_to_adj(maze_arr, goals):
    coord_remap = {}
    for x, y in goals:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                coord_remap[(x + dx, y + dy)] = x, y

    adj = defaultdict(set)
    for x in range(1, maze_arr.shape[0] - 1):
        for y in range(1, maze_arr.shape[1] - 1):
            pt = x, y
            if maze_arr[x, y] == 0 or pt in coord_remap:
                continue
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = x + dx, y + dy
                if maze_arr[nx, ny] == 0:
                    continue
                npt = nx, ny
                if npt in coord_remap:
                    npt = coord_remap[npt]
                adj[pt].add(npt)
                adj[npt].add(pt)
    return adj


def find_goal_edges(maze_adj, goals):
    conns = defaultdict(lambda: defaultdict(set))
    edges = {}

    def dfs(start, curr, visited):
        if curr in visited:
            return
        if curr in goals and curr is not start:
            cvis = deepcopy(visited)
            cvis.remove(start)
            if cvis not in [edges[e] for e in conns[start][curr]]:
                eidx = len(edges)
                edges[eidx] = cvis
                conns[start][curr].add(eidx)
                conns[curr][start].add(eidx)
            return
        visited.add(curr)
        for nbor in maze_adj[curr]:
            dfs(start, nbor, visited)
        visited.remove(curr)

    for goal in goals:
        dfs(goal, goal, set())

    return conns, edges


def eliminate_redundant_edges(connections, edges, edge_conflicts):
    bad_edges = set()
    for g1, econnections in connections.items():
        for g2, conn_edges in econnections.items():
            conn_edges = list(conn_edges)
            for i, edge1 in enumerate(conn_edges):
                for edge2 in conn_edges[i + 1:]:
                    if edges[edge1].issubset(edges[edge2]):
                        bad_edges.add(edge2)
                    if len(edge_conflicts[edge1] ^ edge_conflicts[edge2]) == 0:
                        _, e = max((len(edges[edge1]), edge1),
                                   (len(edges[edge2]), edge2))
                        bad_edges.add(e)
            res = [e for e in conn_edges if e not in bad_edges]
            connections[g1][g2] = sorted(res, key=lambda x: len(edges[x]))
    return connections


def make_conflict_map(edges):
    conflicts = {}
    for edge, points in edges.items():
        conflict = set()
        for edge2, points2 in edges.items():
            if points & points2:
                conflict.add(edge2)
        conflicts[edge] = conflict
    return conflicts


def find_full_path_bfs(connections, edge_conflicts):
    goals = set(connections.keys())
    start = list(connections)[0]

    queue = [(start, set(), set(), set())]
    while len(queue) > 0:
        curr, visited, edges, prohib_edges = queue.pop(0)

        nvisited = deepcopy(visited)
        nvisited.add(curr)
        for nbor, edges in connections[curr].items():
            for edge in edges:
                if edge in prohib_edges:
                    continue
                if nbor is start and goals.issubset(visited):
                    return edges | {edge}
                if nbor in visited:
                    return None
                nedges = deepcopy(edges)
                nedges.add(edge)
                nprohib_edges = deepcopy(prohib_edges)
                nprohib_edges |= edge_conflicts[edge]
                queue.append((nbor, nvisited, nedges, nprohib_edges))
    return None


def find_full_path_dfs(connections, edge_conflicts):
    goals = set(connections.keys())
    start = list(connections)[0]

    def dfs(curr, visited, prohib_edges, depth):
        if curr is start and goals.issubset(visited):
            return set()
        if curr in visited:
            return None

        visited.add(curr)
        for nbor, edges in connections[curr].items():
            for edge in edges:
                if edge in prohib_edges:
                    continue
                conflicts = edge_conflicts[edge]
                res = dfs(nbor, visited, prohib_edges | conflicts, depth + 1)
                if res is not None:
                    return res | {edge}
        visited.remove(curr)

    return dfs(start, set(), set(), 0)


def display_maze_and_path(maze_arr, used_edges, goals,
                          name='maze_solution.png'):
    centers = []
    for x, y in goals:
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                centers.append((x + dx, y + dy))

    disp = maze_arr[..., None].repeat(3, axis=2) * 255
    for edge, color in zip([centers] + list(used_edges),
                           [(255, 0, 0)] + COLORS):
        disp[tuple(zip(*edge))] = np.array(color)

    im = Image.fromarray(disp.astype('uint8'), 'RGB')
    im = im.resize([e * 4 for e in maze_arr.shape])
    im.save(name)


def main():
    maze_arr = load_maze_arr()
    sys.setrecursionlimit(np.sum(maze_arr))

    goals = find_goals(maze_arr)
    maze_adj = convert_to_adj(maze_arr, goals)
    connections, edges = find_goal_edges(maze_adj, goals)
    edge_conflicts = make_conflict_map(edges)
    connections = eliminate_redundant_edges(connections, edges, edge_conflicts)
    used_edges = find_full_path_dfs(connections, edge_conflicts)
    display_maze_and_path(maze_arr, [edges[e] for e in used_edges], goals)


if __name__ == '__main__':
    main()
