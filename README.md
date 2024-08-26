# ML-video-game-analysis-synthetic
A processing of a simplified video game dataset using k-nearest neighbors, and using Synthetic Data Vault and Regression/Classification models to verify synthetic model accuracy

##### data
We use a simplified dataset that lists age, weight, height, gender and gaming preference in terms of RPG, FPS, and so on.

##### approach
We first approach the video game dataset with a simple k-nearest neighbors example, dropping the gaming preference column in an attempt to predict it given different values of age, height, weight and gender.

We then flip this process on an axis by using the same data to predict probable genders, given age, height, weight, and gaming preference.

##### synthetic data generation and verification

We then deviate and use the Synthetic Data Vault to extend the size of the original dataset. This requires installation of the [Synthetic Data Vault packages](https://docs.sdv.dev/sdv/installation). We use this to create a dataset that is 5x the original. 

Using SDV, we create two synthetic sets, a Gaussian Copula version and a CTGAN version.

We want to verify the veracity of the synthetic data, so we use both linear regression (with label conversion to numeric representations) and a classification model (using the gaming preference column as classification labels). 

This classification and linear regression process requires instalation of SKlearn through [scikit-learn](https://scikit-learn.org/stable/install.html).

Pandas and Numpy need to be included but are not listed as they are standard packages. 

