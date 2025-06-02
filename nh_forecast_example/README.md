This directory contains some example data and a configuration file to test my [neuralhydrology](https://github.com/simonmoulds/neuralhydrology.git) fork, specifically additions made in branch [add-forecast-dataset](https://github.com/simonmoulds/neuralhydrology/tree/add-forecast-dataset). 

To run the example code you should set up an environment as detailed in the [neuralhydrology docs](https://neuralhydrology.readthedocs.io/en/latest/usage/quickstart.html#prerequisites). Then follow these steps to install neuralhydrology:

```sh
git clone https://github.com/simonmoulds/neuralhydrology.git
cd neuralhydrology
git branch add-forecast-dataset
pip install -e .
cd ..
```

Then, run `python run-example.py` to train and test the model.  

The `inspect-results.ipynb` notebook makes a plot to visualize the forecast for a single initialization time. 

Please let me know if you encounter any problems. Suggestions and feedback are also welcome! 

Simon Moulds (simon.moulds@ed.ac.uk)