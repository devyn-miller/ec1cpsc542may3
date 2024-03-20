

# CPSC542 Assignment 2

## Student Information

- **Student Name:** Devyn Miller
- **Student ID:** 2409539

## Collaboration

- **In-class collaborators:** Hayden Fargo (not working on the same code, but just bouncing ideas off of one another)

## Resources

- **Resources used:**
   - [The Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/%7Evgg/data/pets/)
   - [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet)
   - Copilot
   - Perplexity

## Data Source

- **Data source:** [oxford_iiit_pet](https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet) `dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)`

## Code Repository

- **GitHub Repository:** [Assignment 2](https://github.com/devyn-miller/the-final-assignment2-cpsc542.git)

# Additional notes below — please read.
1. The u-net model was too large to push to github and can be found at the following link: https://drive.google.com/file/d/1zoP3UmzBmihgdkKlurVVzK1fd9ylbaqz/view?usp=sharing
![Alt text](model.png?raw=true)
2. All of the plots and figures (EDA, grad-CAM, etc) for this project can be found in the Jupyter notebook entitled `main.ipynb`. The report only contains the metrics table and graphs to show model performance.
3. As described in my writeup, I modularized things from an initial Jupyter notebook that contained everything. The png images saved in the `/figures` folder are from that inital ipynb, while those in `main.ipynb` were generated when running the modularized project.
