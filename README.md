# CUDA Galaxy histograms

A school assignment in CUDA that calculates angles between a galaxies with celestial coordinates.

***

The program can be compiled to computers that support CUDA using for instance: 

```bash
nvcc --gpu-architecture=compute_75 -o galaxy ./galaxy.cu
```

***

The program expects the path of two files containing the same amount of galaxies, like:

```bash
./galaxy <path/to/real_galaxy> </path/to/syntethic_galaxy> 
```

Optionally a number can be given as the third argument to print out a few of the results:
```bash
./galaxy <path/to/real_galaxy> </path/to/syntethic_galaxy> 10 
```
Using the `time` command you can time the process.

***

You can also use the [Makefile](./Makefile) and run:

```bash
make build
```

```bash
make run
```

```bash
make clean
```

***

* The GPU configurations can be adjusted in the CUDA file to achieve better performance.  
* The output files will be stored under `./output`.
* The compilation and starting of the program can be adjusted to run on clusters using `srun` or similar.