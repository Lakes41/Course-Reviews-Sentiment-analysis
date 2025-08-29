# Course Review Sentiment Analysis

This project analyzes user reviews for online courses to classify them into 'good' or 'bad' sentiment. The analysis is performed using a Jupyter Notebook (`Main.ipynb`) and involves data cleaning, exploratory data analysis (EDA), and building a machine learning model for sentiment classification.

## Datasets

The project uses two CSV files:

-   `user_courses_review_09_2023.csv`: The primary dataset containing user reviews from September 2023. It includes the following columns: `course_name`, `lecture_name`, `review_rating`, and `review_comment`.
-   `user_courses_review_test_set.csv`: A separate dataset used for validating the final model. It has the same structure as the primary dataset.

## Requirements

To run the analysis in this repository, you will need Python 3 and the following libraries:

-   pandas
-   matplotlib
-   seaborn
-   scikit-learn
-   imbalanced-learn (imblearn)

You can install these dependencies using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

## How to Run

To run the project, you need to have a Jupyter Notebook environment.

1.  Clone this repository to your local machine.
2.  Navigate to the repository's directory in your terminal.
3.  Start Jupyter Notebook by running:
    ```bash
    jupyter notebook
    ```
4.  From the Jupyter interface in your browser, open the `Main.ipynb` file.
5.  You can run the notebook cells sequentially to see the entire analysis process.

## Methodology

The project follows these steps:

1.  **Data Cleaning and Preparation**: The initial dataset is loaded, and missing values are removed. The review ratings are converted to a numeric format, and the review comments are cleaned by converting them to lowercase and removing punctuation.

2.  **Exploratory Data Analysis (EDA)**: The distribution of review ratings is analyzed, revealing a significant class imbalance with a majority of 5-star ratings. The relationship between comment length and review rating is also explored.

3.  **Modeling**:
    -   **Initial Approach (Multi-class Classification)**: An initial attempt to classify reviews into one of five ratings (1-5) using a Multinomial Naive Bayes model was unsuccessful due to the severe class imbalance. The model tended to predict the majority class (5-star rating) for most reviews.
    -   **Binary Classification**: The problem was simplified to a binary classification task by categorizing ratings of 4 or 5 as 'good' and ratings of 1 to 3 as 'bad'.
    -   **Handling Class Imbalance**: The binary classification model still faced a class imbalance. To address this, the minority class ('bad' reviews) was oversampled to match the number of 'good' reviews. This balanced the dataset and significantly improved the model's performance.

4.  **Model Testing**: The final model, a Multinomial Naive Bayes classifier trained on the upsampled binary data, was tested on the `user_courses_review_test_set.csv` to evaluate its performance on unseen data.

## Results

The final model achieved an accuracy of approximately **83%** on the validation dataset. This demonstrates its effectiveness in classifying course reviews into 'good' or 'bad' sentiment, even with the challenges of a highly imbalanced original dataset. The confusion matrix from the validation test shows that the model performs well, particularly in identifying 'good' reviews, while also being able to correctly classify a majority of the 'bad' reviews.
