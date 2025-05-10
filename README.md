# Just In Time Job Shop Scheduling with Sequence Dependent Setup Times

The repository contains the code used in the paper "A Reduced Variable Neighborhood Search for the Just In Time Job Shop Scheduling Problem with Sequence Dependent Setup Times" by Paolo Brandimarte and Edoardo Fadda. 

## Folder Structure

The project is organized as follows:

- **`main_exact.py`** Main script that applies all solvers to a given problem instance.
  
- **`data/`** Contains all the problem instances used in the paper.
  
- **`data_interfaces/`**  Includes functions for reading and parsing data files.
  
- **`instance/`** Defines the Python class used to represent and manage problem instances.
  
- **`logs/`**  Directory where all log files are saved.
  
- **`results/`** Directory where the results of the experiments are printed and stored.

- **`sol_representation/`** Contains visualization tools, such as functions to generate Gantt charts and other graphical representations of solutions.

- **`solvers/`** Includes all code related to the solution methods. It contains two subdirectories:
  
  - **`operators/`** Implements destroy and repair operators (note: these were not used in the final version of the paper).

  - **`just_in_time_job_shop_setup/`** Contains the core solvers, including the exact method, timing models, VNS (Variable Neighborhood Search), etc.


## Citing Us

```Bibtex
@misc{Brandimarte2023,
  author = {Brandimarte, Paolo and Fadda, Edoardo},
  title = {A Reduced Variable Neighborhood Search for the Just In Time Job Shop Scheduling Problem with Sequence Dependent Setup Times},
  year = {2023}
}
```

