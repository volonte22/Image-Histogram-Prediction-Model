Author: Jack Volonte
Date: 9/25/2022
Description: These files create histograms from images, perform a preceptron step-activation algorithim on thsoe histograms, produce various
plots, and tests validation histograms for accuracy. 

File Manifest:
	AurHistogram.txt
	AurUnknowns100.txt
	AurUnknowns300.txt
	histogramMaker.py
	histogramMakerUnknowns.py
	preceptron.py
	README.txt
	validationAurs

THESE FILE NAMES ARE NOT MEANT TO BE RUN - if run, this will ruin the histogram data converted before packaging this homework:
	histogramMaker.py
	histogramMakerUnknowns.py    

USE OF FILES:
	Running the program file - 'preceptron.py' - will execute and run the main program of this homework. This file will perform the 
step activation function, return plots of the weights for each epoch iteration as a line, produce a training error & classification accuracy
vs epoch plot, and write command line the total accuracy of: the validation images, a set of 100 unknown images, and a set of 300 unknown images. 

	Notes when running 'preceptron.py':
		You will be prompted to input Epoch, Learning Rate, and Batch values ( the way I have my data setup, a batch size of 
	769 will be the maximum value for Batches)

TROUBLESHOOTING:
    If you receive an error while running these files, make sure first that your environment has the following imports for:
        matplotlib -> 'pip install matplotlib'
        numpy -> 'pip install numpy'


    If the runnable file doesn't configure, a suggestion would be to install and run on PyCharm's community environment (free to signup and install).
           Make sure to configure with a Python interpreter.

For any questions, please contact:
    volonte22@up.edu
    on Teams,
    volonte22@up.edu
    or by phone,
    503-980-8100

