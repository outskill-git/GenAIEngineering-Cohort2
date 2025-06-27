# Advanced Pandas Homework Assignment

## Overview

This homework assignment is designed to test your understanding of advanced pandas functionalities. You will work with complex data transformations, multi-index operations, performance optimization, and custom functionality extensions.

## Dataset Description

You will be working with three datasets:

1. A sales transactions dataset
2. A customer information dataset
3. A product information dataset

These datasets are designed to simulate real-world data challenges that require advanced pandas techniques to solve efficiently.

## Tasks

### Task 1: Advanced Data Transformation and Reshaping

1. **Pivot and Melt Operations**

   - Convert the sales data from long format to wide format using `pivot` or `pivot_table`
   - Transform the data back to long format using `melt`
   - Create a pivot table that shows monthly sales totals by product category with subtotals

2. **Multi-level Indexing**

   - Create a hierarchical index on the sales data using customer region, product category, and date
   - Perform operations on specific levels of the hierarchy using `xs`
   - Unstacking and restacking levels to reshape the data for different analyses

3. **Advanced GroupBy Operations**
   - Use the `transform` method to normalize sales values within groups
   - Apply multiple aggregation functions simultaneously using `agg`
   - Implement custom aggregation functions
   - Use filter operations to select groups meeting specific criteria

### Task 2: Advanced Merging and Joining

1. **Complex Joins**

   - Perform a three-way join between sales, customer, and product datasets
   - Implement a self-join on the customer dataset to identify hierarchical relationships
   - Use different join types (left, right, inner, outer) and compare the results

2. **Handling Duplicates and Conflicts**

   - Identify and handle duplicate keys when joining datasets
   - Implement a custom conflict resolution strategy when merging data with overlapping columns
   - Create a function that validates the integrity of joined data

3. **Time-Based Joins**
   - Perform asof joins to match records based on timestamps
   - Implement a rolling join that matches each transaction with the most recent customer status update
   - Create a time window join to match events that occurred within a specified time range of each other

### Task 3: Performance Optimization

1. **Memory Usage Optimization**

   - Analyze memory usage of the datasets using `memory_usage(deep=True)`
   - Optimize datatypes to reduce memory footprint (e.g., using categories, smaller integer types)
   - Implement chunking strategies for processing large datasets

2. **Computational Efficiency**

   - Compare the performance of vectorized operations versus apply/iterrows
   - Implement an efficient strategy for updating values based on complex conditions
   - Use `numba` or `swifter` to accelerate pandas operations

3. **Parallel Processing**
   - Split the data into chunks and process in parallel using multiprocessing
   - Implement parallel group operations using `dask` or `modin`
   - Compare execution times between sequential and parallel approaches

### Task 4: Custom Extensions and Advanced Functionality

1. **Custom Accessors**

   - Create a custom pandas accessor that adds domain-specific functionality
   - Implement methods for sales-specific calculations and transformations
   - Add property accessors that compute derived metrics

2. **Method Chaining and Pipelines**

   - Create a data processing pipeline using method chaining
   - Implement custom methods that preserve the DataFrame interface
   - Design a reusable ETL process for the sales data

3. **Advanced Visualization Integration**
   - Create a custom plotting method that generates a dashboard of sales metrics
   - Implement interactive visualizations using plotly
   - Design a function that automatically generates reports with relevant visualizations

### Task 5: Real-world Challenge

Implement a complete solution for a retail analytics scenario:

1. Process raw transaction data to handle missing values, duplicates, and outliers
2. Create customer profiles with calculated metrics (lifetime value, purchase frequency, etc.)
3. Develop a product recommendation engine based on co-purchase patterns
4. Implement time-series forecasting for product sales
5. Create an anomaly detection system for unusual transaction patterns
6. Generate a comprehensive dashboard with actionable insights

## Dataset Generation

Use the following code to generate synthetic datasets for this assignment:

```python
import pandas as pd
import numpy as np
import datetime
from faker import Faker
import uuid

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Define constants
num_customers = 1000
num_products = 200
num_transactions = 50000
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2023, 12, 31)
days_range = (end_date - start_date).days

# Product categories and subcategories
categories = ['Electronics', 'Clothing', 'Home', 'Food', 'Beauty']
subcategories = {
    'Electronics': ['Phones', 'Computers', 'Accessories', 'TVs', 'Audio'],
    'Clothing': ['Men', 'Women', 'Children', 'Shoes', 'Accessories'],
    'Home': ['Furniture', 'Kitchen', 'Decor', 'Bedding', 'Bath'],
    'Food': ['Produce', 'Bakery', 'Dairy', 'Meat', 'Beverages'],
    'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrance', 'Bath & Body']
}

# Regions and countries
regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Oceania']
countries_by_region = {
    'North America': ['USA', 'Canada', 'Mexico'],
    'Europe': ['UK', 'Germany', 'France', 'Italy', 'Spain'],
    'Asia': ['China', 'Japan', 'India', 'South Korea', 'Singapore'],
    'South America': ['Brazil', 'Argentina', 'Colombia', 'Chile', 'Peru'],
    'Africa': ['South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Morocco'],
    'Oceania': ['Australia', 'New Zealand', 'Fiji']
}

# Generate customer data
customer_ids = [str(uuid.uuid4()) for _ in range(num_customers)]
customer_data = []

for customer_id in customer_ids:
    region = np.random.choice(regions)
    country = np.random.choice(countries_by_region[region])
    join_date = start_date + datetime.timedelta(days=np.random.randint(0, days_range))

    customer_data.append({
        'customer_id': customer_id,
        'name': fake.name(),
        'email': fake.email(),
        'phone': fake.phone_number(),
        'region': region,
        'country': country,
        'city': fake.city(),
        'join_date': join_date,
        'tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'], p=[0.5, 0.3, 0.15, 0.05]),
        'is_active': np.random.choice([True, False], p=[0.9, 0.1])
    })

customers_df = pd.DataFrame(customer_data)

# Generate product data
product_ids = [str(uuid.uuid4()) for _ in range(num_products)]
product_data = []

for product_id in product_ids:
    category = np.random.choice(categories)
    subcategory = np.random.choice(subcategories[category])
    launch_date = start_date + datetime.timedelta(days=np.random.randint(0, days_range))

    product_data.append({
        'product_id': product_id,
        'name': fake.word() + ' ' + fake.word().capitalize(),
        'category': category,
        'subcategory': subcategory,
        'price': round(np.random.uniform(10, 1000), 2),
        'cost': round(np.random.uniform(5, 500), 2),
        'weight_kg': round(np.random.uniform(0.1, 20), 2),
        'launch_date': launch_date,
        'is_discontinued': np.random.choice([True, False], p=[0.1, 0.9])
    })

products_df = pd.DataFrame(product_data)

# Generate transaction data
transaction_data = []

for _ in range(num_transactions):
    transaction_date = start_date + datetime.timedelta(days=np.random.randint(0, days_range))
    customer_id = np.random.choice(customer_ids)

    # Each transaction can have 1-5 items
    num_items = np.random.randint(1, 6)

    for _ in range(num_items):
        product_id = np.random.choice(product_ids)
        product_price = products_df.loc[products_df['product_id'] == product_id, 'price'].iloc[0]
        quantity = np.random.randint(1, 5)

        # Apply random discount
        discount_pct = np.random.choice([0, 0, 0, 0.05, 0.1, 0.15, 0.2], p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        price_after_discount = round(product_price * (1 - discount_pct), 2)

        transaction_id = str(uuid.uuid4())

        transaction_data.append({
            'transaction_id': transaction_id,
            'customer_id': customer_id,
            'product_id': product_id,
            'date': transaction_date,
            'quantity': quantity,
            'unit_price': product_price,
            'discount_pct': discount_pct,
            'price_after_discount': price_after_discount,
            'total_amount': round(quantity * price_after_discount, 2),
            'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Bank Transfer']),
            'store_id': np.random.randint(1, 50)
        })

transactions_df = pd.DataFrame(transaction_data)

# Add some missing values and anomalies
transactions_df.loc[np.random.choice(transactions_df.index, size=int(len(transactions_df)*0.01)), 'unit_price'] = np.nan
transactions_df.loc[np.random.choice(transactions_df.index, size=int(len(transactions_df)*0.01)), 'discount_pct'] = np.nan
transactions_df.loc[np.random.choice(transactions_df.index, size=int(len(transactions_df)*0.005)), 'total_amount'] = -1
customers_df.loc[np.random.choice(customers_df.index, size=int(len(customers_df)*0.02)), 'email'] = np.nan
products_df.loc[np.random.choice(products_df.index, size=int(len(products_df)*0.01)), 'price'] = np.nan

# Add some duplicates
dupe_indices = np.random.choice(transactions_df.index, size=int(len(transactions_df)*0.005))
dupes = transactions_df.loc[dupe_indices].copy()
transactions_df = pd.concat([transactions_df, dupes], ignore_index=True)

# Create a time-series for customer status updates
status_updates = []
for customer_id in customer_ids:
    # Generate 1-5 status updates per customer
    num_updates = np.random.randint(1, 6)
    for _ in range(num_updates):
        update_date = start_date + datetime.timedelta(days=np.random.randint(0, days_range))
        status_updates.append({
            'customer_id': customer_id,
            'update_date': update_date,
            'tier': np.random.choice(['Bronze', 'Silver', 'Gold', 'Platinum']),
            'lifetime_value': round(np.random.uniform(0, 10000), 2),
            'credit_score': np.random.randint(300, 851)
        })

status_updates_df = pd.DataFrame(status_updates)
status_updates_df.sort_values('update_date', inplace=True)

# Print sample data
print("Customers sample:")
print(customers_df.head())
print("\nProducts sample:")
print(products_df.head())
print("\nTransactions sample:")
print(transactions_df.head())
print("\nStatus Updates sample:")
print(status_updates_df.head())

# Save to CSV files if needed
customers_df.to_csv('customers.csv', index=False)
products_df.to_csv('products.csv', index=False)
transactions_df.to_csv('transactions.csv', index=False)
status_updates_df.to_csv('status_updates.csv', index=False)
```

## Hints

### Task 1 Hints:

- For pivoting complex data, consider using `pivot_table` instead of `pivot` to handle duplicate values
- When working with hierarchical indices, `swaplevel` and `sortlevel` can help organize your data
- Remember that `transform` preserves the shape of the original DataFrame and aligns the results, while `apply` reduces each group to a single row

### Task 2 Hints:

- When performing complex joins, consider executing them in stages rather than all at once
- For time-based joins, make sure to sort your data by time first
- The `pd.merge_asof` function requires sorted data on the join keys
- Consider using `indicator=True` in merge operations to track the source of each row

### Task 3 Hints:

- Use `pd.to_numeric()` with `downcast` parameter to reduce memory usage
- Consider using `pd.Categorical` for columns with few unique values
- For large datasets, implement a generator that yields chunks of the data
- Vectorized string operations can be performed using `str` accessor methods

### Task 4 Hints:

- Look into `pandas.api.extensions.register_dataframe_accessor` and `pandas.api.extensions.register_series_accessor`
- For method chaining, ensure your custom methods return a DataFrame or Series object
- Consider creating a class that encapsulates your ETL logic for reusability

### Task 5 Hints:

- Break down the complex task into modular components
- Consider using `pd.MultiIndex` to organize multi-dimensional data
- Implement efficient data structures for the recommendation engine
- Use pandas time-series functionality (like `resample` and `rolling`) for forecasting

## Evaluation Criteria

Your submission will be evaluated based on:

1. **Correctness**: Does your code solve the problem accurately?
2. **Efficiency**: Is your solution optimized for performance and memory usage?
3. **Code Quality**: Is your code well-organized, documented, and readable?
4. **Creativity**: Have you implemented innovative approaches to solve complex problems?
5. **Analysis**: Have you provided insightful analysis of the results?

## Submission Requirements

Submit a Jupyter notebook or Python script that includes:

1. Well-commented code for all tasks
2. Explanations of your approach and key decisions
3. Analysis of results with appropriate visualizations
4. Performance metrics for optimized solutions
5. Discussion of limitations and potential improvements
