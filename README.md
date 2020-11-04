# Informative Path Planning Project
## Patrick Phillips

#### Summary
In it's most general form, the Informative Path Planning (IPP) problem is to find a path that maximizes 'information gain' subject to a set of constraints. In this project we are trying to learn an arbitrary 2D field (e.g. a temperature or topography map). The belief state (our current estimate) of the field is modelled using a Gaussian Markov Random Field (GMRF). The belief state is updated with sequential Bayesian conditioning. An ideal metric of 'information gain' would be a measure like mutual information, however, just minimizing the variance of the GMRF proves much more computationally tractable. In this project I run various simulations to explore the effectiveness of different sampling algorithms in solving the IPP problem. An example run is shown in the video below. The top left panel is the true field that is trying to be learned. The top right panel is the belief state (the mean at every point in the GMRF), the bottom left panel is the variance of the GMRF along with the path that the agent has travelled, note that as expected variance is lower near where the agent has explored. 

{% include youtubePlayer.html id="pFDdZPfO63w" %}

#### Code Usage
The directory contains 4 package folders. Alphabetically, the first is the ‘control_algorithms’ package. This package contains the 4 sampling control algorithms PRM, PRM*, RRT, and RRT* in appropriately named files ‘PRM_control.py’, PRM_star_control.py’ and so on. The package also contains a ‘random_walk_control.py’ file, and a ‘control_scripts.py’ file. The ‘control_scripts.py’ file contains the PI^2 algorithm. The ‘control_algorithms’ package also contains the ‘base’ package inside of it. Here there are several local planners and some helper classes. The local planner currently used in all of the algorithms is just the simple dubins_path_planner which uses [Dubin's Paths](https://en.wikipedia.org/wiki/Dubins_path).

The second package is the ‘data’ package. This contains the data collected on the performance of algorithms. New data is automatically collected and stored in this package after every run. Each file has the name of the control algorithm and then the maximum time the algorithm was allowed to run for at each iteration. The shape of the data is a numpy array as follows:

	     [[path_length1, path_length2…........max_path_length]
	     [total variance1, total variance2,..total variance n]
	     [field variance1,field variance2 ...field variance n]
	     [RMSE1, 	RMSE2.....................RMSE n	 ]
	     [calc_time 1, calc_time2.......calc_time_n         ]]

The third package is called ‘gp_scripts.’ This contains the code for the Gaussian Markov Random Field (GMRF) representation of the changing estimated field. It also has the stored precision matrices used for quicker initialization. Whether these are used or precision matrices are reinitialized can be set in the Config file.

The fourth package is called ‘development.’ This contains the same versions of the sampling algorithms but implemented instead for general path planning problems where the goal is to get from some starting configuration to some goal configuration. It also contains some other files developed for understanding and test purposes, such as my own implementation of the sequential GMRF algorithm, none of these are used for the simulation.

The remaining files include the ‘Config.py’ file, the ‘main.py’ file, the ‘plot_data.py’ file the ‘plot_scripts.py’ file and the ‘true_field.py’ file. The Config and main files are probably the most important. The main file runs the project (wow no way!).

The ‘Config.py’ file has all of the knobs to be turned to adjust the simulation, the control algorithms, and the GMRF algorithm. I’ll just go over some important ones and the ones specific to the sampling control algorithms. The simulation_max_dist sets the maximum path length before the algorithm terminates. The max_runtime variable sets the maximum runtime to be allotted for a single control algorithm iteration. The max_curvature sets the maximum turn curvature of the agent. The growth variable here is specific to RRT algorithm details, but might be interesting to try adjusting.

The ‘true_field.py’ file creates the true field which is a set of x, y, z values. This is what the control algorithms are trying to learn as efficiently as possible.

The plot_data file plots data that has been collected. The ‘plot_scripts’ file is used by the main method to plot while the simulation is running. It can be set not to plot anything in the ‘Config.py’ file to speed up data collection. The ways to measure how well the control algorithm is learning the true field is the RMSE (root mean squared error) between the true field and the belief field, and the variance of the belief field. There are two plots of variance vs path length. The first plot that is total variance which includes the variance of these regression coefficients used to learn the mean of the field, as well as the variance of the actual GMRF field. The details of this can be found in Andre Rene Geist’s master thesis.

### Acknowledgments
This code builds off of work by Andre Rene Geist, and was develpoed with supervision by Daniel Deucker at the TUHH in Hamubrg, Germany. Also thanks to the German Academic Exchange program for sponsoring me through the DAAD RISE (Research in Science and Engineering) program.
