# One-Page Business Problem Statement: NYC Taxi ETA Prediction

## 1. Business Objective
Provide highly accurate ETA predictions at ride request time.

## 2. Target Variable & Unit
Target: `trip_duration_minutes` (Derived from timestamps).

## 3. Asymmetric Business Costs
- **Underprediction:** Customer dissatisfaction & missed schedules.
- **Overprediction:** High quoted ETA leads to customer churn.

## 4. Deployment Constraints
Strictly pre-trip features. Post-trip data (`trip_distance`, `fare_amount`) prohibited.
