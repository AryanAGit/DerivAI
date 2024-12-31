This is a sequence to sequence neural network model to perform symbolic differentiation, inspired by the method described by "Deep Learning for Symbolic Mathematics". My data set file wast too large to import, but a similar one can be generated with my dataGen.py program.  
The main.py program is the one to run to train and store the model. It will also output a dictionary, which will be needed to evaluate the model. Pasting the dictionaries in the runAI.py and adding the model's path would allow the user to try the model for themselves 
Creating a smaller, testing dataset would allow you to run testAI.py, to output its accuracy with any dataset and the data on which types of functions it failed.

langFuncs.py, modelFuncs.py, and model.py all define the model and its necessary functions.
