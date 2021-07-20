# Python Fractals
 Python scripts to generate fractals. Initial version use the classic algorithm easily found on internet. Last version (fractalsPyTorch.py) use PyTorch and a custom modified algorithm respect the classical one to elaborate fractals in a very short time. 
A simple video demo is aviable at this [link](https://www.youtube.com/watch?v=Nv-WKpY2qXs) where I show a simple program that create dynamically a Julia set based on the mouse position.

Time to elaborate a 1920x1080 Mandelbrot fractals image with 100 iteration for pixel. Range: x = [-1, 1], y = [-1, 1]:
* Classic algorihtm implementation:~20 seconds.
* Pytorch Implementatin (CPU): &ensp;&ensp;&ensp;  ~3.3 seconds.
* Pytorch Implementatin (CUDA): &ensp;&ensp;~0.2 seconds.
