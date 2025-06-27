# Pandas Homework Assignment

## Overview

This homework assignment is designed to test your understanding of the pandas library in Python. You will work with a dataset of movie information and perform various data manipulation tasks.

## Dataset Description

The dataset contains information about movies including:

- Title
- Release Year
- Genre
- Runtime (minutes)
- Budget (millions USD)
- Box Office (millions USD)
- Director
- Rating (out of 10)
- Country

## Tasks

### Task 1: Basic Data Exploration (Beginner)

1. Load the dataset into a pandas DataFrame.
2. Display the first 5 rows of the DataFrame.
3. Show the shape of the DataFrame (rows and columns).
4. Generate basic descriptive statistics for the numerical columns.
5. Check for missing values in each column.

### Task 2: Data Filtering and Selection (Intermediate)

1. Select all movies released after 2010.
2. Find all movies with a rating higher than 8.0.
3. Create a subset of movies that are either Action or Comedy genres.
4. Identify movies that had a box office greater than twice their budget.
5. List all movies directed by Christopher Nolan or Steven Spielberg.

### Task 3: Data Transformation (Intermediate)

1. Create a new column 'Profit' that calculates the difference between Box Office and Budget.
2. Create a column 'ROI' (Return on Investment) that calculates (Box Office - Budget) / Budget.
3. Create a categorical column 'Length' that classifies movies as 'Short' (< 90 min), 'Medium' (90-120 min), or 'Long' (> 120 min).
4. Create a column 'Decade' that categorizes movies by the decade they were released in (e.g., 1990s, 2000s).
5. Convert the 'Rating' column to a categorical type with bins: 'Poor' (0-4), 'Average' (4-7), 'Excellent' (7-10).

### Task 4: Aggregation and Grouping (Advanced)

1. Calculate the average rating for each genre.
2. Find the highest-grossing movie for each director.
3. Determine the average budget and box office for each decade.
4. Group movies by their country of origin and calculate the mean rating, total budget, and total box office.
5. For each genre, calculate the percentage of movies with ROI > 1 (profitable movies).

### Task 5: Data Visualization (Advanced)

1. Create a bar chart showing the average rating by genre.
2. Generate a scatter plot of Budget vs. Box Office, colored by Rating.
3. Plot the distribution of movie runtimes using a histogram.
4. Create a box plot showing the distribution of ROI by genre.
5. Generate a line plot showing the trend of average budget and box office by year.

## Dataset

```python
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create a sample movie dataset
n_movies = 200

# Generate random data
titles = [f"Movie {i}" for i in range(1, n_movies + 1)]
years = np.random.randint(1990, 2023, n_movies)
genres = np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi', 'Horror', 'Thriller', 'Romance'], n_movies)
runtimes = np.random.randint(75, 180, n_movies)
budgets = np.round(np.random.uniform(5, 250, n_movies), 1)  # In millions USD
box_offices = np.round(budgets * np.random.uniform(0.5, 4, n_movies), 1)  # In millions USD

directors = []
for _ in range(n_movies):
    first_names = ['James', 'Steven', 'Christopher', 'Martin', 'Quentin', 'David', 'Ridley', 'Sofia', 'Greta', 'Kathryn']
    last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 'Taylor']
    directors.append(f"{np.random.choice(first_names)} {np.random.choice(last_names)}")

ratings = np.round(np.random.uniform(3, 9.5, n_movies), 1)
countries = np.random.choice(['USA', 'UK', 'France', 'Japan', 'South Korea', 'India', 'Canada', 'Germany'], n_movies)

# Create the DataFrame
movies_data = pd.DataFrame({
    'Title': titles,
    'Year': years,
    'Genre': genres,
    'Runtime': runtimes,
    'Budget': budgets,
    'BoxOffice': box_offices,
    'Director': directors,
    'Rating': ratings,
    'Country': countries
})

# Introduce some missing values
for col in ['Runtime', 'Budget', 'BoxOffice', 'Rating']:
    missing_indices = np.random.choice(n_movies, size=int(n_movies * 0.05), replace=False)
    movies_data.loc[missing_indices, col] = np.nan

# Display the DataFrame
print(movies_data.head())
```

## Hints

### Task 1 Hints:

- Use `pd.read_csv()` or the sample code provided to create the DataFrame
- Explore methods like `.head()`, `.shape`, `.describe()`, and `.isna().sum()`

### Task 2 Hints:

- Use Boolean indexing with conditions like `df['Year'] > 2010`
- Combine conditions with `&` (and) and `|` (or) operators
- Don't forget to enclose each condition in parentheses when combining them

### Task 3 Hints:

- Use `df['new_column'] = expression` to create new columns
- For categorical columns, consider using `pd.cut()` or `np.select()`
- The formula for ROI is `(BoxOffice - Budget) / Budget`

### Task 4 Hints:

- Utilize `.groupby()` with `.agg()` methods
- For more complex aggregations, consider using `.apply()` or `lambda` functions
- To find maximum values within groups, use `.idxmax()` and then index the original DataFrame

### Task 5 Hints:

- Import matplotlib or seaborn for visualization: `import matplotlib.pyplot as plt` or `import seaborn as sns`
- Group data appropriately before plotting
- Some useful plotting methods: `df.plot()`, `sns.barplot()`, `plt.scatter()`, `sns.boxplot()`
