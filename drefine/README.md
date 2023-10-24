### To run and build the tool go to directory VeriNN/deep_refine

- Instructions to build the tool:
    1. Install blas and lapack:
        1. For ubuntu: `apt install libblas-dev liblapack-dev`

    1. Build drefine:
        1. `goto directory ../VeriNN/deep_refine`
        1. `make`
        
        
* Instructions to run the tool:
    1. Download the [networks](https://drive.google.com/drive/folders/1aV6A8X1naFioCzcamKY3MBX87H_F4uVh?usp=sharing)
    2. `./drefine --help`
    
* Sample command:
   1. `./drefine --network .../tf/mnist/mnist_relu_3_50.tf --dataset-file benchmarks/dataset/mnist/mnist_test.csv --result-file res.txt` 
    
* To run tool on VNNCOMP benchmarks please follow the following sample command: 
    1. First convert the network onnx to tf by using `converter/onnx2tf.py` script. 
    1. `./drefine --network .../tf/mnist/mnist_relu_3_50.tf --vnnlib-prp-file /path/to/vnnlib --result-file res.txt` 






