# uav-sensor-fusion
<img src="docs/drone-simulation.png" width="50%" height="50%">

This is the directory to use the planar drone simulator used to perform state estimation in GPS denied environments but with a known map. The model assumes:
1. two-dimensional world
2. known map
3. Gaussian process and measurement noise
4. pressure measurement
5. height above ground measurement.

With these three metrics, the drone is able to localize itself by solving a nonlinear least squares problem with the [Gauss-Newton algorithm](https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm). The blue/red contour lines represent the gradient that is decended inorder to minimize the cost function. The solution represents the most likely location given different variances on the process and measurement noise.

The file ```nonlinear.ipynb``` can be run in VS Code as a Python Notebook or through https://jupyter.org. The necessary libraries are only numpy, scipy, and matplotlib.