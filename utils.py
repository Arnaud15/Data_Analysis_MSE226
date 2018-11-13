import sklearn as sk
from sklearn import linear_model
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np

def evaluate_cv(X, y, model, cv=10):
    scores = sk.model_selection.cross_val_score(model, X, y, cv)
    return scores


def lassoPath(X, y, coef_names, eps=5e-3, save_fig=True):# the smaller eps is, the longer is the path
	colors = cycle(['b', 'r', 'g', 'c', 'k'])
	X /= X.std(axis=0)  # Standardize data (easier to set the l1_ratio parameter)
	print("Computing regularization path using the lasso...")
	alphas_lasso, coefs_lasso, _ = linear_model.lasso_path(X, y, eps, fit_intercept=False)
	neg_log_alphas_lasso = -np.log10(alphas_lasso)
	plt.figure(1)
	count=0
	for coef_l, c in zip(coefs_lasso, colors):
	    l1 = plt.plot(neg_log_alphas_lasso, coef_l, label=coef_names[count])
	    count+=1
	plt.xlabel('-Log(alpha)')
	plt.ylabel('coefficients')
	plt.title('Lasso')
	plt.legend()
	plt.axis('tight')
	if save_fig:
		plt.savefig('./Saved_Plots/LassoPath.png')
	return

def interaction_terms(my_data, term1, term2, name):
	my_data = my_data.assign(new_feature = my_data.loc[:,term1] * my_data.loc[:,term2])
	my_data.rename(lambda x: name if x=='new_feature' else x, axis=1, inplace=True)
	return my_data

def polynomial_feature(my_data, feature, power):
	my_data = my_data.assign(new_feature = my_data.loc[:, feature] ** power)
	my_data.rename(lambda x: feature + '_power_' + str(power) if x=='new_feature' else x, axis=1, inplace=True)
	return my_data


def lassoChoice(X, y, cv=10):
	print("Computing regularization path using the coordinate descent lasso...")
	model = linear_model.LassoCV(cv=20).fit(X, y)

	# Display results
	m_log_alphas = -np.log10(model.alphas_)

	plt.figure()
	#ymin, ymax = 2300, 3800
	plt.plot(m_log_alphas, model.mse_path_, ':')
	plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
	         label='Average across the folds', linewidth=2)
	plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
	            label='alpha: CV estimate')

	plt.legend()

	plt.xlabel('-log(alpha)')
	plt.ylabel('Mean square error')
	plt.axis('tight')
	#plt.ylim(ymin, ymax)
	return model.alpha_
