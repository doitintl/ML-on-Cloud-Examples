-- Aggregate the data to daily resolution

-- https://stackoverflow.com/questions/41707583/get-total-no-of-days-in-given-month-in-google-bigquery
CREATE TEMP FUNCTION DaysInMonth(d TIMESTAMP) AS (
  32 - EXTRACT(DAY FROM DATE_ADD(DATE_TRUNC(CAST (d AS DATE), MONTH), INTERVAL 31 DAY))
);

WITH monthly_cost_table AS (SELECT 
                      account_name,
                      EXTRACT (MONTH FROM bill_datetime) month,
                      EXTRACT (YEAR FROM bill_datetime) year,
                      SUM(cost) monthly_cost
                      FROM `billing_dataset_example.account_billing_log` 
                      GROUP BY 1,2,3 ORDER BY account_name, year, month),
     daily_costs AS ( SELECT account_name,
                      DaysInMonth( bill_datetime) - EXTRACT (DAY FROM bill_datetime)  day,
                      EXTRACT (MONTH FROM bill_datetime) month,
                      EXTRACT (YEAR FROM bill_datetime) year,
                      SUM(cost) daily_cost
                      FROM `billing_dataset_example.account_billing_log`
                      GROUP BY 1,2,3,4 ORDER BY account_name, year, month, day)
 SELECT daily_costs.*, monthly_cost_table.monthly_cost FROM
 daily_costs  
 LEFT JOIN
 monthly_cost_table ON
 daily_costs.month = monthly_cost_table.month AND
 daily_costs.year = monthly_cost_table.year AND
 daily_costs.account_name = monthly_cost_table.account_name


-- create dataset
WITH
  cumsum_table AS(
  SELECT
    account_name, year, month, day,
    ROUND(SUM(daily_cost) OVER (PARTITION BY account_name, year, month ORDER BY day )) AS cumsum,
    ROUND(AVG(daily_cost) OVER (PARTITION BY account_name, year, month ORDER BY day )) AS mean_daily_cost
  FROM
    `billing_dataset_example.billing_daily_monthly`
  ORDER BY  account_name, year, month, day),
  monthly_cost_table AS (
  SELECT
    account_name,
    EXTRACT(YEAR FROM   bill_datetime ) year,
    EXTRACT(MONTH FROM bill_datetime) month,
    ROUND(SUM(cost)) monthly_cost
  FROM
    `billing_dataset_example.account_billing_log`
  GROUP BY
    1, 2, 3 )
SELECT
  cumsum_table.*,
  monthly_cost_table.monthly_cost
FROM
  cumsum_table
LEFT JOIN
  monthly_cost_table
ON
  monthly_cost_table.account_name = cumsum_table.account_name
  AND monthly_cost_table.year = cumsum_table.year
  AND monthly_cost_table.month = cumsum_table.month
  
  
  -- Train model
  CREATE OR REPLACE MODEL 
  billing_dataset_example.model_linear_regression --model save path
OPTIONS
  ( model_type='linear_reg', -- As of Aug 2018 you can choose between linear regression and logistic regression
    ls_init_learn_rate=.015, 
    l1_reg=0.1,
    l2_reg=0.1,
    data_split_method='seq',
    data_split_col='split_col',
    
    max_iterations=30 -- by default, uses early stopping!
    ) AS
SELECT
    monthly_cost label, -- by naming this field "label" we make it target field
    year split_col,
    -- independent variables:
    cumsum, 
    day,
    mean_daily_cost
FROM
  `billing_dataset_example.linear_regression_dataset` 
  WHERE account_name  = 'DoIT'
  AND YEAR < 2017 -- splitting by year ensures that we don't leak test data to training set


-- Generate predictions
SELECT
  year,
  month,
  day,
  cumsum,
  ROUND(predicted_label) predicted,
  monthly_cost actual_month_cost,
  ROUND(100 * (ABS(predicted_label - monthly_cost) /  monthly_cost),1) abs_err
FROM
  ML.PREDICT(MODEL `billing_dataset_example.model_linear_regression`,
    (
    SELECT
      Year, month,  day,  cumsum,  mean_daily_cost,
      ROUND(monthly_cost) monthly_cost
    FROM
      `billing_dataset_example.linear_regression_dataset`
    WHERE
      account_name = 'DoIT'
      AND YEAR >= 2017))
ORDER BY
  year,
  month,
  day
