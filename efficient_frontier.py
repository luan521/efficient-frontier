import pandas as pd
from tqdm import tqdm
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

if __name__ == '__main__':
    # Load asset data
    assets = ['ITUB3.SA', 'VALE3.SA', 'PETR3.SA', 'ABEV3.SA']
    df = yf.download(assets, start='2020-01-01', progress=False)['Adj Close']

    # Calculate expected returns and covariance matrix
    n_atv = len(assets)
    ind_er = df.resample('Y').last().pct_change().mean()
    cov_matrix = df.pct_change().cov()

    # Initialize lists for portfolio returns, risks, and weights
    p_ret = []
    p_vol = []
    p_pesos = []

    num_portfolios = int(3e4)

    # Simulate random portfolios
    for port in tqdm(range(num_portfolios)):
        pesos = np.random.random(n_atv)
        pesos /= np.sum(pesos)
        p_pesos.append(pesos)
        retornos = np.dot(pesos, ind_er)
        p_ret.append(retornos)
        var = cov_matrix.mul(pesos, axis=0).mul(pesos, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd * np.sqrt(250)
        p_vol.append(ann_sd)

    data = {'Retorno': p_ret, 'Risco': p_vol}
    for counter, symbol in enumerate(df.columns.tolist()):
        data[f'peso {symbol}'] = [w[counter] for w in p_pesos]

    portfolios = pd.DataFrame(data)

    # Function to calculate portfolio risk
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(250)

    # Function to get the efficient frontier points
    target_returns = np.linspace(min(p_ret), max(p_ret), 100)
    efficient_risk = []

    for target_return in target_returns:
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'eq', 'fun': lambda x: np.dot(x, ind_er) - target_return})
        bounds = tuple((0, 1) for _ in range(n_atv))
        result = minimize(portfolio_volatility, n_atv * [1. / n_atv], args=(cov_matrix,),
                        method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_risk.append(result['fun'])

    # Save the efficient frontier Graph
    plt.figure(figsize=(5, 5))
    plt.scatter(portfolios['Risco'], portfolios['Retorno'], c='green', marker='o', s=10, alpha=0.15, label="Portfolios")
    plt.plot(efficient_risk, target_returns, 'b--', linewidth=3, label="Efficient Frontier")
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficient_frontier.jpg')