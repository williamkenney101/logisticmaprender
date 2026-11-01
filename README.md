# Logistic Map Render

## Installation
First run
```bash
pip install pygame numba numpy
```
Then do
```bash
python jp.py
```
## Features
Allows interactive zooming into the map, allowing one to look at specific bifurcation points and chaotic repition.  
Uses Numba to turn calculations into C-level speed, allowing for hundreds of thousands of calculations per frame.  
Density accumulator allows low level bifurcations to be seen while also allowing chaotic ones to be seen.  
Displays change in r and x to see how far you're in.  
Anti aliasing gets rid of pixelation problems and grid-snapping.  
Rainbow chroma effect!  

## What for the future
Taking a look at npmath to push past 64 bit float limits, allowing the simulation to look deeper into the fractal. Currently detail tends to fade as one goes in further.  
Keybind to save a photo (S) to snap pictures/allow for built in video recording.  
Split calculations across multiple cores to further allow for iteration performance, cause frames drop quickly as you zoom in.  
