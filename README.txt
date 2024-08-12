# Project Title

Machine Learning for Loan Default Prediction

## Project Description

This project aims to predict loan defaults using historical data from a German bank. The goal is to build a machine learning model that can accurately identify potential defaulters to minimize financial risk.

## Dataset Description

The dataset, `German_bank.csv`, contains historical loan data for customers, including: 
			checking_balance - Amount of money available in account of customers
			months_loan_duration - Duration since loan taken
			credit_history - credit history of each customers
			purpose - Purpose why loan has been taken
			amount - Amount of loan taken
			savings_balance - Balance in account
			employment_duration - Duration of employment
			percent_of_income - Percentage of monthly income
			years_at_residence - Duration of current residence
			age - Age of customer
			other_credit - Any other credits taken
			housing- Type of housing, rent or own
			existing_loans_count - Existing count of loans
			job - Job type
			dependents - Any dependents on customer
			phone - Having phone or not
			default - Default status (Target column). 
It consists of 1,000 samples with 16 features and one response variable.

## Installation and Setup

To set up the project environment:

1. Install Python 3.8 or above.
2. Clone the repository: `git clone https://github.com/thanhxuan1995/Assignment6.git'
3. Install required packages: `pip install -r requirements.txt`
4. make sure input file stored in '.\data\' folder and '.\data\' folder need to be in your current working folder. 
(ex: if you are working in '.\notebook\' folder, make sure it has '.\notebook\data\credit.csv')

## File Structure

- `data/`: Contains the input dataset file and output files.
- `notebooks/`: Jupyter notebooks for data exploration and analysis.
- `src/`: Python scripts for data preprocessing and model training.
- `reports/`: The final report in PDF format.

## How to Run the Code

To run the analysis, navigate to the `notebooks/` directory and open the Jupyter notebooks in order:

1. `01_data_exploration.ipynb`
2. `02_data_preprocessing.ipynb`
3. `03_model_training.ipynb`

Or:
1. run only 'main.ipynb'

## Results

The key findings are summarized in the `reports/final_report.pdf` file. The best-performing model was the Random Forest classifier with an accuracy of 76%.

## Contact Information

For questions or feedback, please contact Xuan Nguyen at xuan.nguyen.intel@gmail.com

## Acknowledgments

Special thanks to Professors who from University of Arizona, provided me this apportunities to publish my project.
