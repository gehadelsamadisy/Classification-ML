# Part 1

Given the MAGIC gamma telescope dataset. This dataset is generated to simulate registration of high
energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the imaging
technique. The dataset consists of two classes: gammas (signal) and hadrons (background). There are
12332 gamma events and 6688 hadron events.

#### Requirements:

1. **Balance the Dataset**: The dataset was class imbalanced. To balance the dataset, the extra readings for the gamma “g” class were randomly put aside to make both classes equal in size.
2. **Split the Dataset**: The dataset was split randomly so that the training set formed 70%, the validation set 15%, and the test set 15% (the test set was not used while tuning the model parameters).
3. **Apply K-NN Classifier**: The K-NN classifier was applied to the data.
4. **Test Different K Values**: Different K values were applied to get the best results.
5. **Report Metrics**: All of the trained model's accuracy, precision, recall, F1-score, and confusion matrix were reported.

#### Sections:

1. **Import Necessary Libraries**: Imported libraries for data manipulation and machine learning.
2. **Load and Balance the Dataset**: Loaded the dataset and balanced it by downsampling the majority class.
3. **Split the Dataset into Training, Validation, and Test Sets**: Split the balanced dataset into training, validation, and test sets.
4. **Apply the K-Nearest Neighbors Classifier with Different K Values**: Trained and evaluated the KNN classifier with different values of K.
5. **Select the Best K Value Based on F1-Score**: Selected the best K value based on the F1-score from the validation set.
6. **Train the Model with the Best K Value and Evaluate on the Test Set**: Trained the final model with the best K value and evaluated it on the test set.
7. **Analysis of Different K Values and Model Performance**: Analyzed the performance of the model with different K values.
8. **Final Test Set Evaluation**: Evaluated the final model on the test set and interpreted the results.

# Part 2

Given California Houses prices data. This data contains information from the 1990 California census., it
does provide an accessible introductory dataset the basics of regression models.
The data pertains to the houses found in each California district and some summary stats about them
based on the 1990 census data. The columns are as follows; their names are self-explanatory:
• Median House Value: Median house value for households within a block (measured in US
Dollars) [$]
• Median Income: Median income for households within a block of houses (measured in tens of
thousands of US Dollars) [10k$]
• Median Age: Median age of a house within a block; a lower number is a newer building [years]
• Total Rooms: Total number of rooms within a block
• Total Bedrooms: Total number of bedrooms within a block
• Population: Total number of people residing within a block
• Households: Total number of households, a group of people residing within a home unit, for a
block
• Latitude: A measure of how far north a house is; a higher value is farther north [°]
• Longitude: A measure of how far west a house is; a higher value is farther west [°]
• Distance to coast: Distance to the nearest coast point [m]
• Distance to Los Angeles: Distance to the center of Los Angeles [m]
• Distance to San Diego: Distance to the center of San Diego [m]
• Distance to San Jose: Distance to the center of San Jose [m]
• Distance to San Francisco: Distance to the center of San Francisco [m]

This notebook demonstrates the implementation and comparison of Linear Regression, Lasso Regression, and Ridge Regression on the California Housing dataset.

#### Requirements:

1. **Split the Dataset**: The dataset was split randomly so that the training set formed 70%, the validation set 15%, and the test set 15% (the test set was not used while tuning the model parameters).
2. **Apply Regression Models**: Linear, Lasso, and Ridge regression models were applied to the data to predict the median house value.
3. **Report Error Metrics**: Mean Squared Error and Mean Absolute Error were reported for all models.
4. **Comments on Results**: Comments on the results and comparisons between the models were provided.

#### Sections:

1. **Importing Packages and Reading Data**: Imported necessary libraries and loaded the dataset.
2. **Splitting the Data**: Split the dataset into training, validation, and test sets.
3. **Linear Regression**: Trained and evaluated a Linear Regression model, including error calculations (Mean Squared Error and Mean Absolute Error).
4. **Lasso Regression**: Trained and evaluated a Lasso Regression model, including error calculations.
5. **Ridge Regression**: Trained and evaluated a Ridge Regression model, including error calculations.
6. **Models Comparison**: Compared the performance of the three models based on their error metrics.
7. **Comments on Results**: Provided insights and comments on the results of the models.

## Requirements

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn

## Usage

1. Clone the repository.
2. Open the notebooks in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially to execute the code and see the results.

## Dataset

- `ML_Part1.ipynb` uses the `magic04.data` dataset.
- `ML_Part2.ipynb` uses the `California_Houses.csv` dataset.

Make sure to place the datasets in the same directory as the notebooks before running them.
