#include "AnnMaths.h"

vector<vector<double>> AnnMaths::transpose(vector<vector<double>>& x)

{
	vector<vector<double>> y;

	for (int col = 0; col < x[0].size(); ++col)
	{
		y.push_back(vector<double>());
		for (int row = 0; row < x.size(); ++row)
		{
			y[col].push_back(x[row][col]);
		}
	}

	return y;
}

vector<vector<double>> AnnMaths::multiply(vector<vector<double>>& x, vector<vector<double>>& y, bool useSigmoid)
{
	vector<vector<double>> z;

	for (int row = 0; row < x.size(); row++)
	{
		z.push_back(vector<double>());
		for (int col = 0; col < y[0].size(); col++)
		{
			z[row].push_back(0);
			for (int inner = 0; inner < x[0].size(); inner++)
			{
				z[row][col] += x[row][inner] * y[inner][col];
			}
			if (useSigmoid) { z[row][col] = sigmoid(z[row][col]); }
		}
	}

	return z;
}

vector<vector<double>> AnnMaths::getOutputErrors(vector<vector<double>>& targets, vector<vector<double>>& output)
{
	vector<vector<double>> errors;

	for (int neuron = 0; neuron < targets.size(); ++neuron)
	{
		errors.push_back(vector<double>());
		errors[neuron].push_back(/*pow(*/targets[neuron][0] - output[neuron][0]/*, 2)*/);
	}

	return errors;
}

vector<vector<double>> AnnMaths::applyErrors(vector<vector<double>>& errors, vector<vector<double>>& output)
{
	vector<vector<double>> results;

	for (int i = 0; i < errors.size(); ++i)
	{
		results.push_back(vector<double>());
		results[i].push_back(errors[i][0] * output[i][0] * (1 - output[i][0]));
	}

	return results;
}

vector<vector<double>> AnnMaths::applyLearnrate(vector<vector<double>>& x, double lr)
{
	vector<vector<double>> results;

	for (int row = 0; row < x.size(); ++row)
	{
		results.push_back(vector<double>());
		for (int col = 0; col < x[0].size(); ++col)
		{
			results[row].push_back(x[row][col] * lr);
		}
	}

	return results;
}

vector<vector<double>> AnnMaths::add(vector<vector<double>>& x, vector<vector<double>>& y)
{
	vector<vector<double>> results;

	for (int row = 0; row < x.size(); ++row)
	{
		results.push_back(vector<double>());
		for (int col = 0; col < x[0].size(); ++col)
		{
			results[row].push_back(x[row][col] + y[row][col]);
		}
	}

	return results;
}

void AnnMaths::printMatrix(vector<vector<double>>& x)
{
	for (int row = 0; row < x.size(); ++row)
	{
		for (int col = 0; col < x[0].size(); ++col)
		{
			std::cout << x[row][col] << " ";
		}
		std::cout << "\n";
	}
	std::cout << "\n";
}

double AnnMaths::abs(double val)
{
	unsigned long long tmp = *(unsigned long long*)&val;
	tmp &= 0x7FFFFFFFFFFFFFFF;
	return *(double*) &tmp;
}

double AnnMaths::sigmoid(double val)
{
	return 1 / (1 + pow(E, -val));
}

double AnnMaths::fastSigmoid(double val)
{
	return 1 / (1 + abs(val));
}


double AnnMaths::randomVal()
{
	return ((double)rand() / (RAND_MAX) -0.5);
}