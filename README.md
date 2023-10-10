<h1 align='center'><b>Gradient Descent in Mojo 🔥</b></h1>
<p align='center'><sub>
    Implementation of a simple gradient descent problem in Python, Numpy, JAX, C++ (binding with Python) and Mojo.
    My goal here is to make a fair evaluation on the out-of-the-box, raw performance of a tech stack choice. Neither of the implementations is optimal. But what I hope to show is what execution speeds to expect out of the box, the complexity of each implementation and what tech choices have the possibility of squeezing out every bit of performance the hardware has to offer.
	</sub>
</p>


<table>
<tr>
<td><img src="https://github.com/StijnWoestenborghs/gradi-mojo/blob/main/shapes/gifs/circle.gif?raw=true" alt="Circle"></td>
<td><img src="https://github.com/StijnWoestenborghs/gradi-mojo/blob/main/shapes/gifs/sphere.gif?raw=true" alt="Sphere"></td>
<td><img src="https://github.com/StijnWoestenborghs/gradi-mojo/blob/main/shapes/gifs/flame.gif?raw=true" alt="Flame"></td>
<td><img src="https://github.com/StijnWoestenborghs/gradi-mojo/blob/main/shapes/gifs/modular.gif?raw=true" alt="Modular"></td>
</tr>
</table>



## Project Setup



### Prerequisite

System Requirements:
> Mojo v0.4.0  
> Linux: Ubuntu 22.04  
> x86_64 architecture  

Project setup: by running  `make setup`  
> Create virtual environment: `python3 -m venv .venv`  
> Upgrade pip: `. .venv/bin/activate && python -m pip install --upgrade pip`  
> Install project requirements: `. .venv/bin/activate && pip install -r python-requirements.txt`  

### First run
All implementation can be executed by running the **main.mojo** file: `make mo`
> `. .venv/bin/activate && mojo run main.mojo`
- Runs the Mojo implementation
- Python interop to **main.py** > benchmarks function
- Benchmarks Python/Numpy/JAX/C++(binding)
- Python interop to all visualizations

### Configure the optimization problem
From main.mo:  
The shape, optimization target can be adapted by changing the **points** variable. You can choose either:
- A circle of N points (fixed dim = 2)
- A sphere of N points (fixed dim = 3)
- A flame shape (fixed N points)
- A modular shape (fixed N points)

The optimization parameters can be changed:
- dim: Dimensionality of the gradient descent algorithm (visualization support only dim = 2 & 3)
- lr: Learning rate
- niter: Number of iterations (no early stopping is implemented)
- plot: (bool) Generat plots and animations
- run_python: (bool) Run python interop to main.py > benchmarks

### Running the implementations seperately:
Python based implementation can be executed from **main.py**: `make py`
> . .venv/bin/activate && python main.py  

This includes: Python/Numpy/Jax and C++ (binding)

To change the parellelization of the gradient calculations in Mojo: Identify the number of logical CPUs on a Linux system: `nproc` And configure the number of workers in `./mojo/gradient_descent.mojo`
Switching between default and parallel mode can be done by changing how to compute the gradient in gradient_descent function
> compute_gradient[dtype](grad, X, D)  
> compute_gradient_parallel[dtype, nelts](grad, X, D)


### Building the C++ (binding to Python) yourself:  

Both default and parallel (20 workers) C++ binaries are included in the `./cpp/bin` and `./cpp/lib` folder. But you can build the binarie & shared object yourself:  
First unzip the 3rd party eigen-3.4.0.zip library in the `./cpp/include/` folder and compile the C++ code by running `make cpp-build` (g++ build tools installation required).

To change the parellelization of the gradient calculations: Identify the number of logical CPUs on a Linux system: `nproc` And configure the number of workers in `./cpp/src/gradient_descent.cpp`. After building the sharded object (`make cpp-build`). Configure the exect *.so file for the Python binding in `./cpp/binding.py`
> libc = CDLL("cpp/build/lib/gradient_descent_p20.so")


## Gradient Descent Algorithm


$$
D = \begin{bmatrix}
0 & D_{12} &  \cdots & D_{1N} \\
D_{12} & 0 &  \cdots & D_{2N} \\
D_{13} & 0 & \cdots & D_{3N} \\
\vdots & \vdots  & \ddots & \vdots \\
D_{1N} & D_{2N} &  \cdots & 0 \\
\end{bmatrix}
$$



$$
X = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_N \end{bmatrix} = \begin{bmatrix}
x_1 & y_1 & z_1 \\
x_2 & y_2 & z_2 \\
x_3 & y_3 & z_3 \\
\vdots & \vdots & \vdots \\
x_N & y_N & z_N \\
\end{bmatrix}
$$





$$
\text{loss} = \sum_i \sum_j \left( \lVert \mathbf{X}_i - \mathbf{X}_j \rVert^2 - D_{ij}^2 \right)^2
$$




$$
\frac{\partial \text{loss}}{\partial X_i} = \frac{\partial}{\partial X_i} \left( \sum_j \left( \lVert X_i - X_j \rVert^2 - D_{ij}^2 \right)^2 \right) = 2u \frac{du}{\partial X_i}
\\ \text{where } u = \lVert X_i - X_j \rVert^2 - D_{ij}^2
\\= 2 \left( \sum_j \left( \lVert X_i - X_j \rVert^2 - D_{ij}^2 \right) \right) \times 2 (X_i - X_j)
\\= 4 \sum_j \left( \lVert X_i - X_j \rVert^2 - D_{ij}^2 \right) \times (X_i - X_j)
$$


$$
\nabla_X \text{loss} = \begin{bmatrix} 
\frac{\partial \text{loss}}{\partial X_1} \\
\frac{\partial \text{loss}}{\partial X_2} \\
\vdots \\
\frac{\partial \text{loss}}{\partial X_N}
\end{bmatrix}
$$


