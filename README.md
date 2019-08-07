Patrick Phillips
Informative Path Planning Project


This code builds off of the work by Andre Rene Geist. Its purpose is to research and explore the effectiveness of various sampling algorithms in solving the informative path planning problem.

The directory contains 4 package folders. Alphabetically, the first is the ‘control_algorithms’ package. This package contains the 4 sampling control algorithms PRM, PRM*, RRT, and RRT* in appropriately named files ‘PRM_control.py’, PRM_star_control.py’ and so on. The package also contains a ‘random_walk_control.py’ file, and a ‘control_scripts.py’ file. The ‘control_scripts.py’ file contains the PI^2 algorithm. This package also contains the local_planner package and some helper classes. There are several local planners, the one used in all of the algorithms is just the simple dubins_path_planner, which is simply described online.

The second package is the ‘data’ package. This contains the data collected on the performance of algorithms. New data is automatically collected and stored in this package after every run. Each file has the name of the control algorithm and then the maximum time the algorithm was allowed to run for at each iteration. The shape of the data is a numpy array as follows:

	     [[path_length1, path_length2…........max_path_length]
	     [total variance1, total variance2,..total variance n]
	     [field variance1,field variance2 ...field variance n]
	     [RMSE1, 	RMSE2.....................RMSE n	 ]
	     [calc_time 1, calc_time2.......calc_time_n         ]]

The third package is called ‘gp_scripts.’ This contains the code for the Gaussian Markov Random Field (GMRF) representation of the changing estimated field. It also has the stored precision matrices used for quicker initialization. Whether these are used or precision matrices are reinitialized can be set in the Config file.

The fourth package is called ‘path_planning_algorithms.’ This contains the same versions of the sampling algorithms but implemented instead for general path planning problems where the goal is to get from some starting configuration to some goal configuration. This was not the focus of the project, and created as a stepping stone in development.

The remaining files include the ‘Config.py’ file, the ‘main.py’ file, the ‘plot_data.py’ file the ‘plot_scripts.py’ file and the ‘true_field.py’ file. The Config and main files are probably the most important. The main file runs the project (wow no way!).

The ‘Config.py’ file has all of the knobs to be turned to adjust the simulation, the control algorithms, and the GMRF algorithm. I’ll just go over some important ones and the ones specific to the sampling control algorithms. The simulation_max_dist sets the maximum path length before the algorithm terminates. The max_runtime variable sets the max_runtime to be allotted, it is usually actually .05 seconds more than this for the algorithm to complete. The max_curvature sets the maximum turn curvature of the agent. The growth variable here is specific to RRT algorithm details.

The ‘true_field.py’ file creates the true field which is a set of x, y, z values. This is what the control algorithms are trying to learn as efficiently as possible.

The plot_data file plots data that has been collected. The ‘plot_scripts’ file is used by the main method to plot while the simulation is running. It can be set not to plot anything in the ‘Config.py’ file to speed up data collection. The ways to measure how well the control algorithm is learning the true field is the RMSE (root mean squared error) between the true field and the belief field, and the variance of the belief field. There are two plots of variance vs path length. The first plot that is total variance which includes the variance of these regression coefficients used to learn the mean of the field, as well as the variance of the actual GMRF field. The details of this can be found in Andre Rene Geist’s master thesis.
