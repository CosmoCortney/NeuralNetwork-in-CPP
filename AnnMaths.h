#pragma once
#define E 2.718281828459045235
#include<vector>
#include<iostream>
#include<math.h>

using std::vector;
namespace AnnMaths
{
	vector<vector<double>> transpose(vector<vector<double>>& x);
	vector<vector<double>> multiply(vector<vector<double>>& x, vector<vector<double>>& y, bool useSigmoid = false);
	vector<vector<double>> getOutputErrors(vector<vector<double>>& targets, vector<vector<double>>& output);
	vector<vector<double>> applyErrors(vector<vector<double>>& errors, vector<vector<double>>& output);
	vector<vector<double>> applyLearnrate(vector<vector<double>>& x, double lr);
	vector<vector<double>> add(vector<vector<double>>& x, vector<vector<double>>& y);
	void printMatrix(vector<vector<double>>& x);
	double abs(double val);
	double sigmoid(double val);
	double fastSigmoid(double val);
	double randomVal();
};

