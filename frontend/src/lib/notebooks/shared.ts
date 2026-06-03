export const SAMPLE_SOURCE_ROWS = JSON.stringify([
  {
    churn: "Yes",
    contract: "Monthly",
    customer_id: "C-1024",
    monthly_charge: 79.9,
    support_calls: 3,
    tenure_months: 18,
  },
  {
    churn: "No",
    contract: "Annual",
    customer_id: "C-1098",
    monthly_charge: 54.2,
    support_calls: 1,
    tenure_months: 42,
  },
  {
    churn: "No",
    contract: "Annual",
    customer_id: "C-1137",
    monthly_charge: 61.4,
    support_calls: 0,
    tenure_months: 57,
  },
  {
    churn: "Yes",
    contract: "Monthly",
    customer_id: "C-1184",
    monthly_charge: 98.1,
    support_calls: 4,
    tenure_months: 7,
  },
]);

export const SAMPLE_FEATURE_ROWS = JSON.stringify([
  {
    churn_flag: 1,
    contract_monthly: 1,
    monthly_charge_z: 0.92,
    support_calls_z: 0.76,
    tenure_interaction: 1.24,
    tenure_months_z: -0.48,
  },
  {
    churn_flag: 0,
    contract_monthly: 0,
    monthly_charge_z: -0.31,
    support_calls_z: -0.52,
    tenure_interaction: -0.68,
    tenure_months_z: 0.87,
  },
  {
    churn_flag: 0,
    contract_monthly: 0,
    monthly_charge_z: 0.02,
    support_calls_z: -0.95,
    tenure_interaction: 0.75,
    tenure_months_z: 1.12,
  },
  {
    churn_flag: 1,
    contract_monthly: 1,
    monthly_charge_z: 1.38,
    support_calls_z: 1.27,
    tenure_interaction: -1.08,
    tenure_months_z: -1.34,
  },
]);
