# Programming language
python

# OS
Any OS will work, as long as you can install the relevant python version. 

# Environment
Use conda to create the environment and install the dependencies.


# Create and activate conda environment
conda create -n infs7203_env python=3.10
conda activate infs7203_env

# Install dependencies
conda install scikit-learn numpy pandas
conda install -c conda-forge imbalanced-learn

# Run
python main.py