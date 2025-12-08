from Observer_Builder import ObservationBuilder
import numpy as np
import copy
from Env_Builder import *
from parameters import TENTATIVE

import time


class Primal2Observer(ObservationBuilder):
    """
    obs shape: (8 + num_future_steps * obs_size * obs_size )
    map order: poss_map, goal_map, goals_map, obs_map, pathlength_map, blocking_map, deltax_map, deltay_map, astar maps
    """

    def __init__(self, observation_size=11, num_future_steps=3, printTime=False):
        super(Primal2Observer, self).__init__()
        self.observation_size = observation_size
        self.num_future_steps = num_future_steps
        self.NUM_CHANNELS = 8 + self.num_future_steps
        self.printTime = printTime

    def set_world(self, world):
        super().set_env(world)

    def get_next_positions(self, agent_id):
        agent_pos = self.world.getPos(agent_id)
        positions = []
        current_pos = [agent_pos[0], agent_pos[1]]
        next_positions = self.world.blank_env_valid_neighbor(current_pos[0], current_pos[1])
        for position in next_positions:
            if position is not None and position != agent_pos:
                positions.append([position[0], position[1]])
                next_next_positions = self.world.blank_env_valid_neighbor(position[0], position[1])
                for pos in next_next_positions:
                    if pos is not None and pos not in positions and pos != agent_pos:
                        positions.append([pos[0], pos[1]])

        return positions

    def _get(self, agent_id):

        start_time = time.time()

        assert (agent_id > 0)
        agent_pos = self.world.getPos(agent_id)
        top_left = (agent_pos[0] - self.observation_size // 2,
                    agent_pos[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        centre = (self.observation_size - 1) / 2
        obs_shape = (self.observation_size, self.observation_size)

        goal_map = np.zeros(obs_shape)
        poss_map = np.zeros(obs_shape)
        obs_map = np.zeros(obs_shape)
        pathlength_map = np.zeros(obs_shape)
        deltax_map = np.zeros(obs_shape)
        deltay_map = np.zeros(obs_shape)
        blocking_map = np.zeros(obs_shape)

        time1 = time.time() - start_time
        start_time = time.time()


        
            
            
            

        time2 = time.time() - start_time
        start_time = time.time()

        # original layers from PRIMAL1
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):
            for j in range(top_left[1], top_left[1] + self.observation_size):
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of bounds, just treat as an obstacle
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    pathlength_map[i - top_left[0], j - top_left[1]] = -1
                    continue
               
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    pathlength_map[i - top_left[0], j - top_left[1]] = -1
                if self.world.state[i, j] == agent_id:
                    # agent's position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                    # updated_poss_map[i - top_left[0], j - top_left[1]] = 0
                if self.world.goals_map[i, j] == agent_id:
                    # agent's goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append([self.world.state[i, j],i-(top_left[0] + (self.observation_size-1)/2.0),j-(top_left[1] + (self.observation_size-1)/2.0)])
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                    # updated_poss_map[i - top_left[0], j - top_left[1]] = self.world.state[i, j]

                # we can keep this map even if on goal,
                # since observation is computed after the refresh of new distance map
                pathlength_map[i - top_left[0], j - top_left[1]] = self.world.agents[agent_id].distanceMap[i, j]

        time3 = time.time() - start_time
        start_time = time.time()


        dx = self.world.getGoal(agent_id)[0] - agent_pos[0]
        dy = self.world.getGoal(agent_id)[1] - agent_pos[1]
        mag = (dx ** 2 + dy ** 2) ** .5
        if mag != 0:
            dx = dx / mag
            dy = dy / mag
        if mag > 60:
            mag = 60

        time4 = time.time() - start_time
        start_time = time.time()

        current_corridor_id = -1
        current_corridor = self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][1]
        if current_corridor == 1:
            current_corridor_id = \
                self.world.corridor_map[self.world.getPos(agent_id)[0], self.world.getPos(agent_id)[1]][0]

        """
        positions = self.get_next_positions(agent_id)
        for position in positions:
            cell_info = self.world.corridor_map[position[0], position[1]]
            if cell_info[1] == 1:
                corridor_id = cell_info[0]
                if corridor_id != current_corridor_id:
                    if len(self.world.corridors[corridor_id]['EndPoints']) == 1:
                        if [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][0]:
                            blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                                corridor_id,
                                0, agent_id,
                                1)
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][0]:
                        end_point_pos = self.world.corridors[corridor_id]['EndPoints'][0]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            0, agent_id,
                            2)
                    elif [position[0], position[1]] == self.world.corridors[corridor_id]['StoppingPoints'][1]:
                        end_point_pos = self.world.corridors[corridor_id]['EndPoints'][1]
                        deltax_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaX'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        deltay_map[position[0] - top_left[0], position[1] - top_left[1]] = (self.world.corridors[
                            corridor_id]['DeltaY'][(end_point_pos[0], end_point_pos[1])])  # / max(mag, 1)
                        blocking_map[position[0] - top_left[0], position[1] - top_left[1]] = self.get_blocking(
                            corridor_id,
                            1, agent_id,
                            2)
                    else:
                        pass
        """

        time5 = time.time() - start_time
        start_time = time.time()

        free_spaces = list(np.argwhere(pathlength_map > 0))
        distance_list = []
        for arg in free_spaces:
            dist = pathlength_map[arg[0], arg[1]]
            if dist not in distance_list:
                distance_list.append(dist)
        distance_list.sort()
        step_size = (1 / len(distance_list))
        for i in range(self.observation_size):
            for j in range(self.observation_size):
                dist_mag = pathlength_map[i, j]
                if dist_mag > 0:
                    index = distance_list.index(dist_mag)
                    pathlength_map[i, j] = (index + 1) * step_size

        state = np.array([poss_map, goal_map, obs_map, pathlength_map])
   

        time6 = time.time() - start_time
        start_time = time.time()

        return state, [dx, dy, mag],visible_agents, np.array([time1, time2, time3, time4, time5, time6])

    def get_many(self, handles=None):
        observations = {}
        all_visible_agents = {}
        
        if handles is None:
            handles = list(range(1, self.world.num_agents + 1))

        times = np.zeros((1, 6))

        for h in handles:
            state, vector,visible_agents, time = self._get(h)
            observations[h] = [state, vector]
            all_visible_agents[h]=visible_agents
            times += time
        if self.printTime:
            print(times)
        return observations,all_visible_agents

    def get_astar_map(self):
        """

        :return: a dict of 3D np arrays. Each astar_maps[agentID] is a num_future_steps * obs_size * obs_size matrix.
        """

        def get_single_astar_path(distance_map, start_position, path_len):
            """
            :param distance_map:
            :param start_position:
            :param path_len:
            :return: [[(x,y), ...],..] a list of lists of positions from start_position, the length of the return can be
            smaller than num_future_steps. Index of the return: list[step][0-n] = tuple(x, y)
            """

            def get_astar_one_step(position):
                next_astar_cell = []
                h = self.world.state.shape[0]
                w = self.world.state.shape[1]
                for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                    # print(position, direction)
                    new_pos = tuple_plus(position, direction)
                    if 0 < new_pos[0] <= h and 0 < new_pos[1] <= w:
                        if distance_map[new_pos] == distance_map[position] - 1 \
                                and distance_map[new_pos] >= 0:
                            next_astar_cell.append(new_pos)
                return next_astar_cell

            path_counter = 0
            astar_list = [[start_position]]
            while path_counter < path_len:
                last_step_cells = astar_list[-1]
                next_step_cells = []
                for cells_per_step in last_step_cells:
                    new_cell_list = get_astar_one_step(cells_per_step)
                    if not new_cell_list:  # no next step, should be standing on goal
                        astar_list.pop(0)
                        return astar_list
                    next_step_cells.extend(new_cell_list)
                next_step_cells = list(set(next_step_cells))  # remove repeated positions
                astar_list.append(next_step_cells)
                path_counter += 1

            astar_list.pop(0)
            return astar_list

        astar_maps = {}
        for agentID in range(1, self.world.num_agents + 1):
            astar_maps.update(
                {agentID: np.zeros([self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]])})

            distance_map0, start_pos0 = self.world.agents[agentID].distanceMap, self.world.agents[agentID].position

            astar_path = get_single_astar_path(distance_map0, start_pos0, self.num_future_steps)


            if not len(astar_path) == self.num_future_steps:  # this agent reaches its goal during future steps
                distance_map1, start_pos1 = self.world.agents[agentID].next_distanceMap, \
                                            self.world.agents[agentID].goal_pos
                astar_path.extend(
                    get_single_astar_path(distance_map1, start_pos1, self.num_future_steps - len(astar_path)))

            for i in range(self.num_future_steps - len(astar_path)):  # only happen when min_distance not sufficient
                astar_path.extend([[astar_path[-1][-1]]])  # stay at the last pos

            assert len(astar_path) == self.num_future_steps
            for step in range(self.num_future_steps):
                for cell in astar_path[step]:
                    astar_maps[agentID][step, cell[0], cell[1]] = 1

        return np.asarray([astar_maps[i] for i in range(1, self.world.num_agents + 1)])

    def get_tentative_map(self, joint_tentative_actions):
        initial_pos = np.array([self.world.getPos(j + 1) for j in range(self.world.num_agents)])
        move_vectors = np.stack([joint_tentative_actions.get(j + 1, np.zeros((self.num_future_steps, 2))) for j in range(self.world.num_agents)])
        current_pos = initial_pos.copy()
        all_final_pos = np.zeros((self.num_future_steps, self.world.num_agents, 2), dtype=np.int32)
        for t in range(self.num_future_steps):
            predicted_pos = current_pos + move_vectors[:, t, :]
            predicted_pos = predicted_pos.astype(np.int32)

            y_pred = predicted_pos[:, 0]
            x_pred = predicted_pos[:, 1]

            in_bounds = (y_pred >= 0) & (y_pred < self.world.state.shape[0]) & \
                    (x_pred >= 0) & (x_pred < self.world.state.shape[1])
            
            is_not_obstacle = np.ones(self.world.num_agents, dtype=bool)

            valid_indices = np.where(in_bounds)[0] #in_boundsがfalseのところは考える必要なし
            if len(valid_indices) > 0:
                is_not_obstacle[valid_indices] = self.world.state[y_pred[valid_indices], x_pred[valid_indices]] != -1
          
            is_valid_move = in_bounds & is_not_obstacle

            final_pos_t = current_pos.copy() #止まるやつはそのまま
            final_pos_t[is_valid_move] = predicted_pos[is_valid_move]

            current_pos = final_pos_t
            all_final_pos[t] = final_pos_t

        final_pos=all_final_pos.transpose(1,0,2)

        all_tentative_maps = np.zeros((self.world.num_agents, self.num_future_steps, self.world.state.shape[0], self.world.state.shape[1]), dtype=np.int32)
        y_coords = final_pos[..., 0]
        x_coords = final_pos[..., 1]

        agent_indices = np.arange(self.world.num_agents)[:, np.newaxis]
        agent_indices = np.repeat(agent_indices, self.num_future_steps, axis=1) 

       
        step_indices = np.arange(self.num_future_steps)[np.newaxis, :]
        step_indices = np.repeat(step_indices, self.world.num_agents, axis=0)

        all_tentative_maps[agent_indices, step_indices, y_coords, x_coords] = 1
        return all_tentative_maps


    def get_blocking(self, corridor_id, reverse, agent_id, dead_end):
        def get_last_pos(agentID, position):
            history_list = copy.deepcopy(self.world.agents[agentID].position_history)
            history_list.reverse()
            assert (history_list[0] == self.world.getPos(agentID))
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        positions_to_check = copy.deepcopy(self.world.corridors[corridor_id]['Positions'])
        if reverse:
            positions_to_check.reverse()
        idx = -1
        for position in positions_to_check:
            idx += 1
            state = self.world.state[position[0], position[1]]
            if state > 0 and state != agent_id:
                if dead_end == 1:
                    return 1
                if idx == 0:
                    return 1
                last_pos = get_last_pos(state, position)
                if last_pos == None:
                    return 1
                if idx == len(positions_to_check) - 1:
                    if last_pos != positions_to_check[idx - 1]:
                        return 1
                    break
                if last_pos == positions_to_check[idx + 1]:
                    return 1
        if dead_end == 2:
            if not reverse:
                other_endpoint = self.world.corridors[corridor_id]['EndPoints'][1]
            else:
                other_endpoint = self.world.corridors[corridor_id]['EndPoints'][0]
            state_endpoint = self.world.state[other_endpoint[0], other_endpoint[1]]
            if state_endpoint > 0 and state_endpoint != agent_id:
                return -1
        return 0


if __name__ == "__main__":
    pass
