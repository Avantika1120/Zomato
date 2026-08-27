# Zomato Delivery ETA Optimization

**Predicting delivery time more accurately to reduce late orders and improve customer experience**

`Python` `Pandas` `Scikit-learn` `Random Forest` `Linear Regression` `Feature Engineering` `Cross-Validation` `Business Analytics`

---

## Business Problem

Delivery ETAs directly shape customer trust. When predicted delivery times are too optimistic, late orders can increase cancellations, support tickets, compensation costs, and customer churn.

This project asks:

> **Can historical delivery data be used to produce more accurate ETAs and reduce the risk of severe delivery delays?**

The analysis connects model performance to a business question: whether better ETA reliability could improve repeat-order behavior and create measurable economic value.

---

## Dataset

The project uses order-level delivery data with **45,584 original records and 20 columns**. After cleaning and preparation, approximately **39,600 records** and **16 predictors** were used for modeling.

Key information includes:

- Restaurant and delivery coordinates
- Road traffic density
- Weather conditions
- Festival indicator
- Vehicle condition
- Multiple-delivery load
- Order date and time
- Actual delivery time

### Target

`time_taken (minutes)` — continuous delivery time, framed as a supervised regression problem.

---

## Analytical Workflow

### 1. Data Cleaning

Handled missing timestamps and categorical values, removed unreliable fields, filtered extreme trips above 80 km, and checked for corrupt or duplicate records.

### 2. Feature Engineering

Created business-aware predictors such as:

- `distance_km` using Haversine distance
- `hour_of_day`, `day_of_week`, `is_weekend`
- `hour_sin`, `hour_cos` for cyclical time behavior
- operational-load features from multiple deliveries and vehicle condition
- external-condition features from traffic, weather, and festivals

A delayed-order flag was also engineered by comparing actual versus benchmark-predicted delivery times. Orders more than **10.2 minutes above expected time** were treated as delayed.

### 3. Modeling

Two approaches were compared:

- **Linear Regression** — transparent baseline for directional interpretation
- **Random Forest Regressor (500 trees)** — captures nonlinear interactions among traffic, distance, weather, timing, and operational load

Models were evaluated using a hold-out test set and K-Fold cross-validation.

---

## Model Performance

| Model | Setting | MAE ↓ | RMSE ↓ | R² ↑ |
|---|---|---:|---:|---:|
| Linear Regression | Hold-out | 5.71 | 7.14 | 0.42 |
| Linear Regression | K-Fold OOS | 5.68 | 7.10 | 0.43 |
| Random Forest | In-sample | 5.06 | 6.47 | 0.53 |
| **Random Forest** | **K-Fold OOS** | **4.999** | **6.39** | **0.536** |

The Random Forest produced the strongest out-of-sample performance and reduced severe high-residual late-delivery cases.

---

## What Drives Delivery Time?

Model explainability highlighted several operational drivers:

- Festival orders: approximately **+11.9 minutes** in the linear model
- Multiple deliveries: approximately **+4.05 minutes**
- Distance: approximately **+0.39 minutes per km**
- Traffic jams: positive delay effect
- Low traffic and sunny weather: shorter expected delivery times

Random Forest feature importance similarly emphasized distance, multiple deliveries, traffic, weather, hour of day, and festivals.

---

## Business Impact

The project goes beyond prediction accuracy by translating ETA improvement into potential business value.

A value framework was built around average order value, reduction in delay probability, and the estimated relationship between delivery reliability and order frequency.

Under the project assumptions, the modeled uplift was approximately **$0.0089 per order**, with a bootstrapped **95% confidence interval of $0.0083–$0.0092 per order**.

This estimate should be treated as a business hypothesis rather than a guaranteed production outcome; the next step is controlled experimentation.

---

## Deployment Recommendation

A production version would score an order when the customer is viewing a restaurant or checkout page using only information available at order creation.

**Input:** location, distance, traffic, weather, time, festival status, operational load  
**Output:** predicted delivery ETA

Before full rollout, I would validate the business effect through an **A/B test** comparing the existing ETA logic with the new model on:

- ETA error
- delayed-order rate
- cancellation rate
- customer-support contacts
- repeat-order frequency

---

## Repository Contents

- `Zomato Data Science - Project.ipynb` — complete analysis and modeling workflow
- `graphs.pdf` — supporting model and business visualizations
- `README.md` — recruiter-friendly project summary

---

## Key Takeaway

This project demonstrates how machine learning can be connected to an operational decision rather than treated as a standalone prediction exercise. The strongest model reduced ETA error while the business analysis translated improved reliability into a testable customer-retention and revenue hypothesis.
