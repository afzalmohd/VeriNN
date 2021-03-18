
--we use modifier.py to modify the constraints we got from ERAN to some understandable format 

	we get input_modified.txt from input_unmodified.txt (input layer constraints)
	we get internal_modified.txt from internal_unmodified.txt (internal layer constraints)
	we get output_modified.txt from output_unmodified.txt (output layer constraints)

--z3_format_converter.py convert modified constrainsts to z3 understandable format 
  as well as solve some necessary constrainsts and returns a model. model have eta
  values corresp 

--z3 take constraints and solve some related constraints and returns a model

--newz3.py is a python file which contains main function and will read the arguments on command line 
  and call appropriate functions.

--arguments : dataset, network name, etc

Command : python3 newz3.py mnist_relu_3_50.tf mnist output_modified.txt internal_modified.txt
