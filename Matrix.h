#pragma once
#include<iostream>
#include<vector>
#include<cmath>
#include<chrono>
#include<memory>

class Matrix
{
private:
	unsigned int m_nRows = 0;
	unsigned int m_nCols = 0;

public:
	std::vector<std::vector<float>> m_data;			     //data made puplic for direct/fast access
	Matrix(int nRows, int nCols, bool randInit = false); //if randInt is true Matrix will be initialized with random values
	Matrix() {}
	unsigned int get_numCols() const;					 //returns amount of columns
	unsigned int get_numRows() const;					 //returns amount of rows
	float sumCol(const unsigned int col) const;			 //sums all column values at the given row
	float sumRow(const unsigned int row) const;			 //sums all row values at the given column
	Matrix transpose() const;							 //returns a transposed copy
	void print() const;									 //print this Matrix
	Matrix operator+(Matrix& instance) const;			 //add matrices
	Matrix operator-(Matrix& instance) const;			 //add subtract
	Matrix operator*(Matrix& instance) const;			 //multiply matrices
	Matrix operator*(Matrix&& instance) const;			 //multiply matrices
	Matrix operator*(float val) const;					 //multiply each value by val
	Matrix operator/(Matrix& instance) const;			 //divide matrices
	Matrix dot(Matrix& instance) const;					 //create dot product
};