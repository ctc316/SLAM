import random
import numpy as np
from MapUtils.bresenham2D import bresenham2D
from MapUtils.MapUtils import mapCorrelation, getMapCellsFromRay


class Slam:
    def __init__(self, n_data, lidar_angles, startingAngle, noise, log_probability, ocpy_thresh, empty_thresh, res=0.1, grid_size=1000, iters_start_map=0, n_particles=100):
        self.ocpy_prob = log_probability
        self.empty_prob = -log_probability
        self.ocpy_thresh = ocpy_thresh
        self.empty_thresh = empty_thresh
        
        self.res = res  # meter
        self.grid_size = grid_size
        self.grid_orig = self.grid_size // 2
        self.grid = np.zeros((self.grid_size, self.grid_size))
    
        self.n_data = n_data
        self.odometry_loc = np.zeros((n_data, 2))
        self.particle_loc = np.zeros((n_data, 2))
        
        self.iters = 0
        self.iters_start_map = iters_start_map
        
        # lidar
        self.lidar_angles = lidar_angles 
        self.startingAng = startingAngle
        
        # particle filter
        self.n_particles = n_particles  # number of particles
        self.W = np.ones(n_particles) / n_particles  # initialize same prior
        self.particles = np.zeros((n_particles, 3))  # N, (x, y, theta)
        self.noise = noise  # (x, y, theta)
        self.prev_pose = None  # np.array[x, y, theta]
        
    
    def update(self, x, y, theta, scan, tilted):
        x = (x / self.res).astype(int) 
        y = (y / self.res).astype(int)
        
        if self.prev_pose is None:
            self.prev_pose = np.array([x, y, theta])
        
        cor_mle = self.__particle_predict(x, y, theta, scan)
        px, py, ptheta = self.__particle_update(x, y, theta, scan, cor_mle)
        
        x += self.grid_orig
        y += self.grid_orig
        px += self.grid_orig
        py += self.grid_orig
        
        if self.iters >= self.iters_start_map and not tilted:
            self.__grid_update(px, py, ptheta, scan)  
           
        self.odometry_loc[self.iters] = np.array([x, y])
        self.particle_loc[self.iters] = np.array([px, py])
          
        self.iters += 1
    
    
    def __particle_predict(self, x, y, theta, scan):
        if self.prev_pose is None:
            self.prev_pose = np.array([x, y, theta])
            return
        
        # particles
        pose = np.array([x, y, theta])
        pose_diff = pose - self.prev_pose
        self.prev_pose = pose
        self.particles += pose_diff + np.random.randn(self.n_particles, 3) * self.noise
        self.particles[:, 2] %= 2 * np.pi
         
        # index of x and y coordinate
        x_im = np.arange(self.grid_size)
        y_im = np.arange(self.grid_size)

        # 5x5 window
        xs = np.arange(-2 * self.res, 3 * self.res, self.res)
        ys = np.arange(-2 * self.res, 3 * self.res, self.res)
        
        # temp grid for computing map correlation
        grid = np.zeros_like(self.grid)
        grid[self.grid > self.ocpy_thresh] = 1
        grid[self.grid < self.empty_thresh] = -1

        # iterate N particles
        cor_mle = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            # pose in world frame
            theta = (self.lidar_angles + self.particles[i][2]).reshape((-1, 1))
            px = scan * np.cos(theta) / self.res + self.grid_size // 2
            py = scan * np.sin(theta) / self.res + self.grid_size // 2
            
            # map correlation
            map_corr = mapCorrelation(grid, x_im, y_im, 
                                      np.vstack((px, py)), 
                                      self.particles[i][0] + xs, 
                                      self.particles[i][1] + ys)
            cor_mle[i] = np.max(map_corr)
        
        return cor_mle
        
    
    def __particle_update(self, x, y, theta, scan, cor_mle):
        cor_mle *= self.W
        self.W = self.__softmax(cor_mle)

        best_idx = np.argmax(self.W)
        px, py, ptheta = self.particles[best_idx]
        
        n_eff = 1 / (self.W ** 2).sum()
        if n_eff < 0.85 * self.n_particles:
            idx = self.__stratified_resample(self.W)
            self.particles[:] = self.particles[idx]
            self.W.fill(1.0 / self.n_particles)
        
        return int(px), int(py), ptheta 
    
        
    def __grid_update(self, x, y, theta, scan):
        cx, cy = self.__pol2cart(scan, theta)
        cx = (cx / self.res).astype(int)
        cy = (cy / self.res).astype(int)
        
        x_occ = cx + x
        y_occ = cy + y
        wall_set = np.zeros_like(self.grid) 
        for i in range(len(x_occ)):
            if 0 <= x_occ[i] < self.grid_size and 0 <= y_occ[i] < self.grid_size:
                wall_set[x_occ[i], y_occ[i]] = 1
 
        empty_set = np.zeros_like(self.grid)
        for i in range(len(x_occ)):
            line = np.array(bresenham2D(0, 0, cx[i], cy[i])).astype(int)
            line[0] += x
            line[1] += y
            for j in range(len(line[0]) - 1):
                if 0 <= line[0][j] < self.grid_size and 0 <= line[1][j] < self.grid_size:
                    empty_set[line[0][j], line[1][j]] = 1

        self.grid[wall_set == 1] += self.ocpy_prob
        self.grid[empty_set == 1] += self.empty_prob

        # pixel value
        self.grid[self.grid > 120] = 120
        self.grid[self.grid < -120] = -120
        
    
    def getResults(self):
        slam_map = np.zeros_like(self.grid)
        slam_map[self.grid > self.ocpy_thresh] = 1
        slam_map[self.grid < self.empty_thresh] = 0.65
        return slam_map, self.particle_loc, self.odometry_loc
    
  
    def __pol2cart(self, radius, theta):
        theta = np.arange(self.startingAng, -self.startingAng, -2 * self.startingAng / len(radius)) + theta
        x, y = radius * np.cos(theta), radius * np.sin(theta)
        return x, y
    
    
    def __softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    
    
    def __stratified_resample(self, weights):
        N = len(weights)
        rng = (np.random.rand(N) + range(N)) / N
        cum_w = np.cumsum(weights)
        result = []
        i, j = 0, 0
        while i < N:
            if rng[i] < cum_w[j]:
                result.append(j)
                i += 1
            else:
                j += 1
        return np.array(result)