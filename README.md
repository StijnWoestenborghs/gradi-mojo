<h1 align='center'><b>Gradient Descent in Mojo 🔥</b></h1>
<p align='center'><sub>
    Implementation of a simple gradient descent problem in Python, Numpy, JAX, C++ (binding with Python) and Mojo.
    My goal here is to make a fair evaluation on the out-of-the-box, raw performance of a tech stack choice. Neither of the implementations is optimal. But what I hope to show is what execution speeds to expect out of the box, the complexity of each implementation and what tech choices have the possibility of squeezing out every bit of performance the hardware has to offer.
	</sub>
</p>


![Circle](https://github.com/StijnWoestenborghs/gradi-mojo/blob/gifs/shapes/gifs/circle.gif?raw=true)

![Circle](https://github.com/StijnWoestenborghs/gradi-mojo/blob/gifs/shapes/gifs/sphere.gif?raw=true)

![Circle](https://github.com/StijnWoestenborghs/gradi-mojo/blob/gifs/shapes/gifs/flame.gif?raw=true)

![Circle](https://github.com/StijnWoestenborghs/gradi-mojo/blob/gifs/shapes/gifs/modular.gif?raw=true)



## Project setup

### Prerequisite

> - Linux: 
>    `make setup`


If you want to build the cpp binary for yourself

http://eigen.tuxfamily.org/

unzip the library



parallel gradient calculation

To get the maximum available number of threads on your Linux machine, you typically look for the number of logical CPUs, since each logical CPU can typically handle one thread at a time. 
Here are several ways to determine the number of logical CPUs on a Linux system:
nproc


