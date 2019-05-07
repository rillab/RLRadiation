import numpy as np
import warnings
from collections import deque, defaultdict

class RadEnv2D_New:
    '''The radiation simulation environment for reinforcement learning'''

    def __init__(self, x_range, y_range, background=25):
        """Initialize the radiation simulation environment
        
        It defines the simulated area's x_range, y_range, and background radiation level.
        The simulated area is x_range by y_range.

        Parameters
        ----------
        x_range: int, the x-direction length of the simulation environment.
        y_range: int, the y-direction length of the simulation environment.
        background: int, the background radiation level of the simulation environment.

        Returns
        -------
        None.
        """
        self.x_range = x_range
        self.y_range = y_range
        self.background_radiation = background
   

    def _generateMap(self, buildings):
        """Set the map of the experiment area
        
        The simulated area is x_range by y_range. In order to denote the edge of the area,
        we used [x_range+2, y_range+2] matrix to represent the map:
        
        Example: x_range = 8, y_range = 6.
       
            11111111
            10000001
            10000001
            11110001
            10000001
            11111111
            
            '1' means it is edge or building.
            '0' means it is movable space.
        
        Parameters
        ----------
        buildings: list(set(2-tuple)). A list of the 'building' input for the 
            self.addBuilding function. It denotes all the buildings in the map.
        
        Returns
        -------
        None.
        """
        # Initialize/reset map
        self.map = np.zeros([self.x_range+2, self.y_range+2])
        self.map[:,0] = 1
        self.map[:,-1] = 1
        self.map[0,:] = 1
        self.map[-1,:] = 1
        # Add buildings
        if buildings:
            for bd in buildings:
                self._addBuilding(bd)
        

    def _bfsShortestDistance(self):
        """Calculate the shortest distance map for the simulation environment

        Use BFS search to calculate the shortest distance between each position 
        to the source, this function should be called after the placement of the source

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.shortest_distance = np.ones([self.x_range+2, self.y_range+2])*(-1)
        visited = set()
        que = deque()
        que.append((round(self.source_x), round(self.source_y)))
        parents = defaultdict()
        parents[(round(self.source_x), round(self.source_y))] = None
        while que:
            current = que.pop()
            if current not in visited:
                if parents[current]==None:
                    self.shortest_distance[current[0], current[1]] = 0
                else:
                    p = parents[current]
                    self.shortest_distance[current[0], current[1]] = self.shortest_distance[p[0], p[1]] + 1
                for child in [
                    (current[0]+1, current[1]), 
                    (current[0]-1, current[1]), 
                    (current[0], current[1]+1),
                    (current[0], current[1]-1)]:
                    if self.map[child[0], child[1]]!=1:
                        parents[(child[0], child[1])] = current
                        que.appendleft((child[0], child[1]))
                visited.add(current)


    def _bfsDistanceReward(self, old_x, old_y):
        """ Calculate the reward according to the distance between source and detector

        The reward is 0.5 if the detector moves closer to the source.
        The reward is -1.5 if the detector moves further to the source.

        Parameters
        ----------
        old_x: int, the x coordinate position for previous step.
        old_y: int, the y coordinate position for previous step.

        Returns
        -------
        reward: float, the reward for current action.
        """
        previous_distance = self.shortest_distance[old_x, old_y]
        current_distance = self.shortest_distance[self.current_x, self.current_y]
        if previous_distance <= current_distance:
            reward = -1.5    
        else:
            reward = 0.5
        return reward                   
        
    
    def _addBuilding(self, building):
        """Add buildings to the simulation environment
        
        Add buildings to the self.map. 1 denotes buildings, and map edges;
        0 denotes movable positions. 2 denotes current detector position.
        
        Parameters
        ----------
        self: the class pointer.
        
        building: a set of 2-tuples denoting the building occupied block. 
            e.g.: set((1,2),(1,3),(1,4)) means there is a wall from (1,2) to (1,4).
            
        Returns
        -------
        None.
        """
        if building:
            for b in building:
                if self.map[b[0],b[1]] == 1:
                    raise ValueError('The place is already occupied by buildings')
                if self.map[b[0],b[1]] == 2:
                    raise ValueError('This place is currently occupied by the detector.')
                self.map[b[0],b[1]] = 1
          

    def _isBlocked(self, p1, p2):
        """Test if two point p1 (viewer) and p2 (source) are blocked by buildings.
        
        Definition of blocking:
            (1) Draw a line from p1 to p2. If this line cross any building, then
            p1 and p2 are blocked. 
            (2) If p1 and p2 are the same point, then they are not blocked. 
            (3) If either p1 or p2 is the building block, then they are blocked.
            
        Parameters
        ----------
        p1: list[x,y]
        p2: list[x,y]
        
        Returns
        -------
        Boolean. True means blocked, False means not blocked.
        """
        total_steps = np.ceil(max(abs(p1[0]-p2[0]), abs(p1[1]-p2[1]))).astype(int)
        x_steps = self._partitionDistance(p1[0], p2[0], total_steps).astype(int)
        y_steps = self._partitionDistance(p1[1], p2[1], total_steps).astype(int)

        if total_steps>0:
            for i in range(len(x_steps)):
                if self.map[x_steps[i], y_steps[i]] == 1:
                    return True
        else: return False
        return False
    

    def _partitionDistance(self, x,y,steps):
        """Partition the distance [x,y] into #steps segments.
        
        e.g.: partitionDistance(1,3,2) = np.array([1,2,3])
        
        Parameters
        ----------
        x: int, the starting point.
        y: int, the ending point.
        steps: int, number of segments we want to partition.
        
        Returns
        -------
        res: numpy.array(int), the partitioned segments.
        """
        if x == y:
            return np.array([x]*(steps+1))
        else:
            d = (y-x)/steps
            res = np.array([x+i*d for i in range(steps+1)])
        return res
    

    def _putSource(self, x,y, i):
        """Place the radiation source into the simulation environment.

        Parameters
        ----------
        x: float, x coordinate of the source position
        y: float, y coordinate of the source position
        i: int, the intensity of the radiation source

        Returns
        -------
        None.
        """
        self.source_x = x
        self.source_y = y
        self.source_intensity = i
        self.radiation_map = np.ones([self.x_range+2, self.y_range+2])*(-1)
        for i in range(1, self.x_range+1):
            for j in range(1, self.y_range+1):
                # If this position is blocked by the buildings
                if self._isBlocked([i,j], [self.source_x, self.source_y]):
                    self.radiation_map[i,j] = self.background_radiation
                # If this position is not blocked by the buildings
                else:
                    d_squared = (i-self.source_x)**2+(j-self.source_y)**2
                    # radiation map for (float, float) source.
                    scaled_i = self.source_intensity/d_squared + self.background_radiation
                    self.radiation_map[i,j] = scaled_i

    
    def _normalizeCNNState(self, state):
        """Rescale the state input to [0,1] for each feature. 
        
        Parameters
        ----------
        state: nunmpy.ndarray, [WORLD_SIZE, WORLD_SIZE, n_feature]
        
        Returns
        -------
        normalized_state: nunmpy.ndarray, [WORLD_SIZE, WORLD_SIZE, n_feature]
        """
        n_features = state.shape[2]
        normalized_state = state.copy()
        
        if state.dtype != 'float64':
            warnings.warn('Caution:the data type of the state is not correct.')
        
        for i in range(n_features-1): # the last feature is the map, which shouldn't be normalized
            if state[:,:,i].max() !=0:
                normalized_state[:,:,i] = state[:,:,i]/state[:,:,i].max()
        return normalized_state
    
    
    def _moveLeft(self):
        """Move the agent one step left (x reduces one).
        
        Parameters
        ----------
        None

        Returns
        -------
        Boolean. True if the movement is successful, false otherwise. 
        """
        if self.map[self.current_x - 1, self.current_y] == 1:
            return False
        else:
            self.current_x = self.current_x - 1
        return True
    

    def _moveRight(self):
        """Move the agent one step right (x adds one).
        
        Parameters
        ----------
        None

        Returns
        -------
        Boolean. True if the movement is successful, false otherwise. 
        """
        if self.map[self.current_x + 1, self.current_y] == 1:
            return False
        else:
            self.current_x = self.current_x + 1
        return True
    

    def _moveDown(self):
        """Move the agent one step down (y reduces one).
        
        Parameters
        ----------
        None

        Returns
        -------
        Boolean. True if the movement is successful, false otherwise. 
        """
        if self.map[self.current_x, self.current_y - 1] == 1:
            return False
        else:
            self.current_y = self.current_y - 1
        return True
    

    def _moveUp(self):
        """Move the agent one step up (y adds one).
        
        Parameters
        ----------
        None

        Returns
        -------
        Boolean. True if the movement is successful, false otherwise. 
        """
        if self.map[self.current_x, self.current_y + 1] == 1:
            return False
        else:
            self.current_y = self.current_y + 1
        return True
    

    def _getMeasurement(self, old_x, old_y):
        """ Get a radiation measurement at the agent's current location

        Parameters
        ----------
        old_x: None or int. Denote the previous step's x position.
        old_y: None or int. Denote the previous step's y position.

        Returns
        -------
        poisson_measure: int, a radiation measurement.
        """
        if hasattr(self, 'source_x') == False and hasattr(self, 'source_x_list') == False:
            print('no source in the environment')
            return
        #Inverse-Squared_law
        poisson_mean = self.radiation_map[self.current_x, self.current_y]
        poisson_measure = np.random.poisson(lam=poisson_mean)
        # remember previous measurements 
        self.radiation_mean[self.current_x, self.current_y]  = \
            (self.radiation_mean[self.current_x, self.current_y]*self.measurement_number[self.current_x, self.current_y]+poisson_measure)/\
            (self.measurement_number[self.current_x, self.current_y]+1)
        self.measurement_number[self.current_x, self.current_y] +=1
        # update current location in the map. '2' is used to denote the current
        # position of the detector.
        if old_x != None and old_y != None:
            # reset previous step position's indicator to 0. 
            self.map[old_x, old_y] = 0
        # set the current step position's indicator to 2.
        self.map[self.current_x, self.current_y] = 2
        return poisson_measure
    

    def stepCNN(self,action):
        """Move one step in the simulation environment.

        Parameters
        ----------
        action: int, encode the action the agent chooses.
        
        Returns
        -------
        normalized_return_state: nunmpy.ndarray, [WORLD_SIZE, WORLD_SIZE, n_feature]
        reward: float, reward for this step's action
        terminate: boolean, if the detector finds the source or not after this step's movement

        """
        # record previous step's position
        old_x = self.current_x
        old_y = self.current_y
        # make one movement
        if action == 0:
            succeed = self._moveLeft()
        elif action == 1:
            succeed = self._moveRight()
        elif action == 2:
            succeed = self._moveUp()
        elif action == 3:
            succeed = self._moveDown()
        # get a new measurement
        _ = self._getMeasurement(old_x, old_y)
        # Obtain reward
        reward = self._bfsDistanceReward(old_x, old_y)
        # If the detector has found the source
        if (np.abs(self.current_x-self.source_x)<=0.5) and (np.abs(self.current_y-self.source_y)<=0.5):
            fake_return_state = np.zeros([self.x_range+2, self.y_range+2, 3])
            normalized_return_state = fake_return_state # this value will not be actually used 
            terminate = True
        # If the detector has not found the source
        else:
            return_state = np.array(
                [self.measurement_number,
                 self.radiation_mean,
                 self.map])
            return_state = np.moveaxis(return_state, 0, -1)
            normalized_return_state = self._normalizeCNNState(return_state)
            terminate = False
        return normalized_return_state, reward, terminate


    def resetCNN(self, buildings, current_location, source_location, source_intensity='random'):
        '''Reset the game for the reinforcement learning algorithm
        
        Paremeters
        ----------
        self: class object
        
        buildings: list(set(2-tuple)). A list of the 'building' input for the 
            self.addBuilding function. It denotes all the buildings in the map.
        
        current_location: two-tuple: (current_x, current_y). Denote the agent's current location.
            If it is 'random', the algorithm will randomly sample the current_location.
            
        source_location: two_tuple: (source_x, source_y). Denote the radiation source's location.
            If it is 'random', the algorithm will randomly sample the source location.

        source_intensity: int or 'random'. Denote the radiation source's intensity.
            If it is 'random', the algorithm will uniformly sample the source intensity from (3000,7000).
        
        Returns
        -------
        normalized_return_state: nunmpy.ndarray, [WORLD_SIZE, WORLD_SIZE, n_feature]
        '''
        old_x = None
        old_y = None
        if hasattr(self, 'current_x'):
            old_x = self.current_x
            old_y = self.current_y
        # reset the world's map
        self._generateMap(buildings)
        # reset the current detector position
        if current_location == 'random':
            while True:
                prop_current_x = np.random.randint(1, self.x_range+1)
                prop_current_y = np.random.randint(1, self.y_range+1)
                if self.map[prop_current_x, prop_current_y] == 0:
                    self.current_x = prop_current_x
                    self.current_y = prop_current_y
                    break
        else:
            if self.map[current_location[0], current_location[1]] == 1:
                raise ValueError('Trying to set the agent on buildings or out of map positions')
            else:
                self.current_x = current_location[0]
                self.current_y = current_location[1]
        # reset the measurement_number matrix and radiation_mean matrix
        self.measurement_number = np.zeros([self.x_range+2, self.y_range+2])
        self.radiation_mean = np.zeros([self.x_range+2, self.y_range+2])
        # reset the radiation source position
        if source_location == 'random':
            while True:
                prop_source_x = np.random.randint(1, self.x_range)+np.random.rand()
                prop_source_y = np.random.randint(1, self.y_range)+np.random.rand()
                # Avoid putting the source in building blocks.
                if self.map[round(prop_source_x), round(prop_source_y)] == 0:
                    self._putSource(
                        prop_source_x, 
                        prop_source_y,
                        np.random.randint(3000,7000))
                    break
        else:
            if self.map[round(source_location[0]), round(source_location[1])] == 1:
                raise ValueError('Trying to set the source location on buildings or out of map positions')
            else:
                if source_intensity != 'random':
                    intensity = source_intensity
                else:
                    intensity = np.random.randint(3000,7000)
                self._putSource(
                    source_location[0],
                    source_location[1],
                    intensity)
        # Obtain the shortest distance from each block to the source
        self._bfsShortestDistance()
        # get the first measurement in the reset environment
        _ = self._getMeasurement(old_x, old_y)
        # prepare return state
        return_state = np.array(
            [self.measurement_number,
             self.radiation_mean,
             self.map])
        return_state = np.moveaxis(return_state, 0, -1)
        normalized_return_state = self._normalizeCNNState(return_state)
        return normalized_return_state
    
    # ----------------------Multiple source related functions-----------------------------------
    # Currently, the multiple source module only supports testing scenarios. It doesn't support training.
    
    def _putSource_multiple(self, x_list,y_list, i_list):
        """Place the radiation source into the simulation environment.

        Parameters
        ----------
        x_list: list of float, x coordinates of the source position
        y_list: list of float, y coordinates of the source position
        i_list: list of int, the intensities of the radiation source

        Returns
        -------
        None.
        """
        self.source_x_list = x_list
        self.source_y_list = y_list
        self.source_intensity_list = i_list
        self.radiation_map = np.ones([self.x_range+2, self.y_range+2])*(-1)
        for i in range(1, self.x_range+1):
            for j in range(1, self.y_range+1):
                # Calculate the radiation intensity for each position
                radiation_intensity = self.background_radiation
                # iterate over all sources
                for k in range(len(self.source_x_list)):
                    # If this position is not blocked by the buildings
                    if not self._isBlocked([i,j], [self.source_x_list[k], self.source_y_list[k]]):
                        d_squared = (i-self.source_x_list[k])**2+(j-self.source_y_list[k])**2
                        # radiation map for (float, float) source.
                        scaled_i = self.source_intensity_list[k]/d_squared
                        radiation_intensity = radiation_intensity + scaled_i
                self.radiation_map[i,j] = radiation_intensity
                
    def resetCNN_multiple_source(self, buildings, current_location, source_location_list, source_intensity_list='random'):
        '''Reset the game for the reinforcement learning algorithm
        
        Paremeters
        ----------
        self: class object
        
        buildings: list(set(2-tuple)). A list of the 'building' input for the 
            self.addBuilding function. It denotes all the buildings in the map.
        
        current_location: two-tuple: (current_x, current_y). Denote the agent's current location.
            If it is 'random', the algorithm will randomly sample the current_location.
            
        source_location_list: tuple of two lists: (source_x_list, source_y_list). Denote the radiation sources' locations.
            If it is 'random', the algorithm will randomly sample the source location.

        source_intensity_list: list of int or 'random'. Denote the radiation sources' intensities.
            If it is 'random', the algorithm will uniformly sample the source intensity from (3000,7000).
        
        Returns
        -------
        normalized_return_state: nunmpy.ndarray, [WORLD_SIZE, WORLD_SIZE, n_feature]
        '''
        old_x = None
        old_y = None
        if hasattr(self, 'current_x'):
            old_x = self.current_x
            old_y = self.current_y
        # reset the world's map
        self._generateMap(buildings)
        # reset the current detector position
        if current_location == 'random':
            while True:
                prop_current_x = np.random.randint(1, self.x_range+1)
                prop_current_y = np.random.randint(1, self.y_range+1)
                if self.map[prop_current_x, prop_current_y] == 0:
                    self.current_x = prop_current_x
                    self.current_y = prop_current_y
                    break
        else:
            if self.map[current_location[0], current_location[1]] == 1:
                raise ValueError('Trying to set the agent on buildings or out of map positions')
            else:
                self.current_x = current_location[0]
                self.current_y = current_location[1]
        # reset the measurement_number matrix and radiation_mean matrix
        self.measurement_number = np.zeros([self.x_range+2, self.y_range+2])
        self.radiation_mean = np.zeros([self.x_range+2, self.y_range+2])
        # reset the radiation source position
        if source_location_list == 'random':
            while True:
                prop_source_x = np.random.randint(1, self.x_range)+np.random.rand()
                prop_source_y = np.random.randint(1, self.y_range)+np.random.rand()
                # Avoid putting the source in building blocks.
                if self.map[round(prop_source_x), round(prop_source_y)] == 0:
                    self._putSource(
                        prop_source_x, 
                        prop_source_y,
                        np.random.randint(3000,7000))
                    break
        else:
            for k in range(len(source_location_list[0])):
                if self.map[round(source_location_list[0][k]), round(source_location_list[1][k])] == 1:
                    raise ValueError('Trying to set the source location on buildings or out of map positions')
            if source_intensity_list != 'random':
                intensity_list = source_intensity_list
            else:
                intensity_list = np.random.randint(3000,7000, size=len(source_location_list[0]))
            self._putSource_multiple(
                source_location_list[0],
                source_location_list[1],
                intensity_list)
        # Obtain the shortest distance from each block to the source
        # self._bfsShortestDistance()
        # get the first measurement in the reset environment
        _ = self._getMeasurement(old_x, old_y)
        # prepare return state
        return_state = np.array(
            [self.measurement_number,
             self.radiation_mean,
             self.map])
        return_state = np.moveaxis(return_state, 0, -1)
        normalized_return_state = self._normalizeCNNState(return_state)
        return normalized_return_state
    
    def stepCNN_multiple_source(self,action):
        """Move one step in the simulation environment with more than one source.

        Parameters
        ----------
        action: int, encode the action the agent chooses.
        
        Returns
        -------
        normalized_return_state: nunmpy.ndarray, [WORLD_SIZE, WORLD_SIZE, n_feature]
        reward: float, reward for this step's action
        terminate: boolean, if the detector finds the source or not after this step's movement

        """
        # record previous step's position
        old_x = self.current_x
        old_y = self.current_y
        # make one movement
        if action == 0:
            succeed = self._moveLeft()
        elif action == 1:
            succeed = self._moveRight()
        elif action == 2:
            succeed = self._moveUp()
        elif action == 3:
            succeed = self._moveDown()
        # get a new measurement
        _ = self._getMeasurement(old_x, old_y)

        return_state = np.array(
            [self.measurement_number,
             self.radiation_mean,
             self.map])
        reward = 0
        terminate = False
        
        return_state = np.moveaxis(return_state, 0, -1)
        #normalized_return_state = self._normalizeCNNState(return_state)
        terminate = False
        return return_state, reward, terminate
