import numpy as np
import sys
import os
import onnx
from onnx import numpy_helper

def my_onnx2nnet(onnxFile, inputMins=None, inputMaxes=None, means=None, ranges=None, nnetFile="", inputName="", outputName=""):
    '''
    Write a .nnet file from an onnx file
    Args:
        onnxFile: (string) Path to onnx file
        inputMins: (list) optional, Minimum values for each neural network input.
        inputMaxes: (list) optional, Maximum values for each neural network output.
        means: (list) optional, Mean value for each input and value for mean of all outputs, used for normalization
        ranges: (list) optional, Range value for each input and value for range of all outputs, used for normalization
        inputName: (string) optional, Name of operation corresponding to input.
        outputName: (string) optional, Name of operation corresponding to output.
    '''
    
    if nnetFile=="":
        nnetFile = onnxFile[:-4] + 'nnet'

    model = onnx.load(onnxFile)
    graph = model.graph
    
    if not inputName:
        #print(graph.input)
        #print(type(graph.input))
        #print(graph.input[0])
        #assert len(graph.input)==1
        inputName = graph.input[0].name
    #print("Check.........................")
    if not outputName:
        #print(graph.output)
        assert len(graph.output)==1
        outputName = graph.output[0].name
    
    # Search through nodes until we find the inputName.
    # Accumulate the weight matrices and bias vectors into lists.
    # Continue through the network until we reach outputName.
    # This assumes that the network is "frozen", and the model uses initializers to set weight and bias array values.
    weights = {}
    biases = {}
    relus = {}
    layer_index = 0
    
    # Loop through nodes in graph
    #print(type(graph.node))
    is_followed_matmul = False
    for node in graph.node: 
        #print(node.input, node.output[0], node.op_type)
        if inputName in node.input:
            if node.op_type in ["Sub","Div","Flatten"]:
                inputName = node.output[0]
                is_followed_matmul = False
            elif node.op_type == "Gemm" or node.op_type == "MatMul":
                if len(node.input)==3:
                    for val in node.input:
                        if 'weight' in val:
                            weight_name = val
                        elif 'bias' in val:
                            bias_name = val
                    weights[layer_index] = [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name==weight_name]
                    biases[layer_index] = [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name==bias_name]
                    layer_index += 1
                elif len(node.input) == 2:
                    for val in node.input:
                        if 'MatMul' in val:
                            weight_name = val
                    
                    weights[layer_index] = [numpy_helper.to_array(inits).T for inits in graph.initializer if inits.name==weight_name]
                else:
                    assert 0

                is_followed_matmul = True
                inputName = node.output[0]
            
            elif node.op_type == "Add":
                assert len(node.input) == 2 and is_followed_matmul
                for val in node.input:
                        if 'Add' in val:
                            bias_name = val
                biases[layer_index] = [numpy_helper.to_array(inits) for inits in graph.initializer if inits.name==bias_name]
                layer_index += 1
                is_followed_matmul = False
                inputName = node.output[0]
            elif node.op_type == "Relu":
                relus[layer_index-1] = "ReLU"
                inputName = node.output[0]
            else:
                print("Node operation type %s not supported!"%node.op_type)
                weights = []
                biases=[]
                break
            if outputName == inputName:
                break

    # Check if the weights and biases were extracted correctly from the graph
    if outputName==inputName and len(weights)>0 and len(weights)==len(biases):
        
        inputSize = weights[0][0].shape[1]
        
        # Default values for input bounds and normalization constants
        # if inputMins is None: inputMins = inputSize*[np.finfo(np.float32).min]
        # if inputMaxes is None: inputMaxes = inputSize*[np.finfo(np.float32).max]
        # if means is None: means = (inputSize+1)*[0.0]
        # if ranges is None: ranges = (inputSize+1)*[1.0]
        f = open(nnetFile, 'w')
        print(weights.keys())
        print(biases.keys())
        print(relus.keys())
        for key in weights.keys():
            layer_str = str(weights.get(key)[0].tolist())+"\n"
            layer_str += str(biases.get(key)[0].tolist())+"\n"
            op = relus.get(key)
            if op == None:
                op = "Gemm"
            layer_str = op+"\n"+layer_str
            f.write(layer_str)
        f.write("\n")
        f.close()

            
            
        # Print statements
        print("Converted ONNX model at %s"%onnxFile)
        print("    to an NNet model at %s"%nnetFile)
        
        # Write NNet file
        #writeNNet(weights,biases,inputMins,inputMaxes,means,ranges,nnetFile)
        
    # Something went wrong, so don't write the NNet file
    else:
        print("Could not write NNet file!")

   
if __name__ == '__main__':
    # Read user inputs and run onnx2nnet function
    # If non-default values of input bounds and normalization constants are needed, 
    # this function should be run from a script instead of the command line
    if len(sys.argv)>1:
        print("WARNING: Using the default values of input bounds and normalization constants")
        onnxFile = sys.argv[1]
        if len(sys.argv)>2:
            nnetFile = sys.argv[2]
            #onnx2nnet(onnxFile,nnetFile=nnetFile)
            my_onnx2nnet(onnxFile, nnetFile=nnetFile)
        else: my_onnx2nnet(onnxFile)
    else:
        print("Need to specify which ONNX file to convert to .nnet!")

    # onnx_dir = '/home/u1411251/Documents/Phd/tools/networks/onnx/cifar10'
    # tf_dir = '/home/u1411251/Documents/Phd/tools/networks/tf/cifar10'
    # for file in os.listdir(onnx_dir):
    #     file_without_extension = os.path.splitext(str(file))[0]
    #     tf_file_path = tf_dir+'/'+file_without_extension+'.tf'
    #     onnx_file_path = onnx_dir+'/'+str(file)
    #     print(str(file))
    #     my_onnx2nnet(onnx_file_path, nnetFile=tf_file_path)
