from scipy.optimize import minimize

# Define the objective function for portfolio optimization
def objective(weights, returns, cov_matrix, target_return):
    portfolio_return = np.dot(weights, returns)
    portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
    return portfolio_variance - target_return * portfolio_return

# Set the expected returns and covariance matrix for the assets
returns = np.array([0.05, 0.08, 0.10])  # Example returns for three assets
cov_matrix = np.array([[0.04, 0.02, 0.01],
                       [0.02, 0.06, 0.03],
                       [0.01, 0.03, 0.05]])  # Example covariance matrix

# Set the target return for the portfolio
target_return = 0.08

# Set the initial weights for the optimization
initial_weights = np.array([1/3, 1/3, 1/3])

# Perform portfolio optimization
result = minimize(objective, initial_weights, args=(returns, cov_matrix, target_return),
                  bounds=[(0, 1), (0, 1), (0, 1)], constraints=({'type': 'eq', 'fun': lambda weights: 1 - np.sum(weights)}))

# Get the optimized weights
optimized_weights = result.x

# Print the optimized weights
print("Optimized Weights:", optimized_weights)