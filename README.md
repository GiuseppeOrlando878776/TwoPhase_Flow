# TwoPhase_Flow

This repository implements a base class **TwoPhaseFlow** into the file "TwoPhaseFlow.py" which provides a general framework for immiscible, isothermal and incompressible two-phase flows. \
The solution of the Navier-Stokes(NS) equations is coupled with a level-set method which can be stabilized through SUPG or Interior Penalty (IP) method.
For what concerns the NS part no stabilization term is added (therefore low Reynolds numbers have to be employed) and the solution can be obtained either solving
a unique linear system (Standard method) or using a projection scheme (ICT method).\
The level-set function has to be reinitialized in order to keep some properties and a suitable step is added for this purpose;
moreover a Conservative level-set method (CLSM) has also be implemented. \
Finally, two derived classes (**BubbleMove** and **RayleighTaylor**) have been implemented in order to simulate two test-cases, whereas some useful functions
(like regualarized Heaviside or regualrized Delta) and definition of the boundaries are respectively in the files "Auxiliary_Functions.py" and "Boundary_Definition.py".

## Rayleigh-Taylor instability
The file "Rayleigh_Taylor.py" contains a derived class **RayleighTaylor** in order to simulate a non-dimensional version of the Rayleigh-Taylor instability. \
The fluid is initially at rest and the interface can be perturbed either with a 'cos' or 'tanh'. \
It is possible to pass directly the adimensional parameters (Atwood number and Reynolds number) or, as an alternative, to set the values of density and compute the adimensional
parameters from that

## Rising bubble
The file "Bubble_move.py" contains a derived class **BubbleMove** in order to simulate a dimensional version of the rising bubble in a viscous flow. \
The fluid is initially at rest and the initial location of the interface is: \
![equation](http://www.sciweavers.org/tex2img.php?eq=%20%5Csqrt%7B%28x-x_0%29%5E2%20%2B%20%28y-y_0%29%5E2%7D%20-%20r&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

## Configuration file
The configuration file is a text file that contains a user’s settings for a particular problem to be solved. Its parsing is implemented in "My_Parameters.py"
This section describes the options available:
- **Problem**: 'Bubble' or 'RT' ('Bubble' by default)
- **Saving_Directiory**: directory where to save the solution ('Sim' by default)
- **Saving_Frequency**: how often current state has to be saved (50 by default)
- **Log_Level**: level of verbosity for DOLFIN output (21 by default)
- **Reference_Dimensionalization**: 'Dimensional' or 'Non_Dimensional' setting ('Dimensional by default')
- **Polynomial_degree**: degree of polynomial for NS P<sub>k+1</sub>/P<sub>k</sub> with k ≥ 1 (1 by default)
- **Number_vertices_x**: number of points in x direction (80 by default)
- **Number_vertices_y**: number of points in y direction (160 by default)
- **Interface_Thickness**: value of the thickness of interface for non-conservative level-set method (0.025 by default)
- **Reinit_Type**: choice of level-set policy between 'Non_Conservative_Hyperbolic' or 'Conservative' ('Non_Conservative_Hyperbolic' by default)
- **Reinitialization_Frequency**: how often reinitialization has to be performed (1 by default)
- **Tolerance_recon**: tolerance for reinitialization step (10<sup>-4</sup> by default)
- **Maximum_subiters_recon**: maximum number of iterations for reinitialization step (10 by default)
- **Stabilization_Type**: choice of stabilization between 'None','IP' and 'SUPG' ('SUPG' by default)
- **Stabilization_Parameter**: parameter for stabilization (0.01 by default)
- **NS_Procedure**: way of solving NS between 'Standard' and 'ICT' ('ICT' by default)
- **Settings_Type**: way of reading data for RT instability between 'Physical' and 'Parameters' ('Physical' by default)
- **Interface_Perturbation_RT**: initial perturbation for RT instability between 'Cos' and 'Tanh' ('Cos' by default)
- **Time_step**: time-step to be employed for the equations
- **End_time**: final time for the simulation
- **Gravity**: modulus of acceleration of gravity
- **Surface_tension**: value of surface tension coefficient
- **Lighter_density**: value of density for lighter fluid
- **Heavier_density**: value of density for heavier fluid
- **Viscosity_lighter_fluid**: value of viscosity for lighter fluid
- **Viscosity_heavier_fluid**: value of viscosity for heavier fluid
- **Base**: width of the computational box
- **Height**: height of the computational box
- **x_center**: initial x coordinate for the centre of the bubble (only for 'Bubble' problem)
- **y_center**: initial y coordinate for the centre of the bubble (only for 'Bubble' problem)
- **Radius**: initial radius of the bubble (only for 'Bubble' problem)

The options without a default value must be supplied. \
After the name of the option, there must be a space, then an '=' sign and then another space before specifying the value desired. \
Please refer to .cfg files in the repository for some clarifying examples

## Execution of the code
In order to run the code, it is necessary a text file with the options for the specific problem. The syntax to run a simulation is
```
python3 main.py your_config_file
```
In case no file is passed in input, it will try with 'test.cfg' and, if it is no present, a default one will be created. \
The code is able to support parallel computations; therefore if MPI is available you can run
```
mpirun -n nproc python3 main.py your_config_file
```
where nproc is the number of processes you want to employ.

## Post-processing
For the rising bubble there are some interesting benchmark quantities whose value is saved throughout the simulation. \
The Post-processing of this data can be performed either in MATLAB or Python respectively with the file "post-process.m"
and "post-process.py"
The MATLAB version can be executed as follows:
```
matlab -r "post_process(your_config_file)"
```
with the name of the file between quotes, whereas the Python version can be run in this way:
```
python3 post_process.py your_config_file
```
It is fundamental that the files are in the same directory of the configuration file.\
Even in this case, if no input argument is supplied, the code will try to analyse 'test.cfg'
