#include "Matrix.h"

Matrix::Matrix(int nRows, int nCols, bool randInit)
{
	for (int row = 0; row < nRows; ++row)
	{
		m_data.push_back(std::vector<float>());
		for (int col = 0; col < nCols; ++col)
		{
			m_data[row].push_back(randInit ? (float)rand() / (RAND_MAX) : 0.0f);
		}
	}

	m_nRows = nRows;
	m_nCols = nCols;
}

float Matrix::sumCol(const unsigned int col) const
{
	float sum = 0;
	for (int row = 0; row < m_nRows; ++row)
	{
		sum += m_data[row][col];
	}

	return sum;
}

float Matrix::sumRow(const unsigned int row) const
{
	float sum = 0;
	for (int col = 0; col < m_nRows; ++col)
	{
		sum += m_data[row][col];
	}

	return sum;
}

Matrix Matrix::transpose() const
{
	Matrix dataT(m_nCols, m_nRows);

	for (int row = 0; row < m_nCols; ++row)
	{
		for (int col = 0; col < m_nRows; ++col)
		{
			dataT.m_data[row][col] = m_data[col][row];
		}
	}

	return dataT;
}

void Matrix::print() const
{
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			std::cout << m_data[row][col] << ((col == m_nCols - 1) ? "\n" : ", ");
		}
	}
}

Matrix Matrix::operator+(Matrix& instance) const
{
	Matrix result(m_nRows, m_nCols);
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			result.m_data[row][col] = m_data[row][col] + instance.m_data[row][col];
		}
	}
	return result;
}

Matrix Matrix::operator-(Matrix& instance) const
{
	Matrix result(m_nRows, m_nCols);
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			result.m_data[row][col] = m_data[row][col] - instance.m_data[row][col];
		}
	}
	return result;
}

Matrix Matrix::operator*(Matrix& instance) const
{
	Matrix result(m_nRows, m_nCols);
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			result.m_data[row][col] = m_data[row][col] * instance.m_data[row][col];
		}
	}
	return result;
}

Matrix Matrix::operator*(Matrix&& instance) const
{
	Matrix result(m_nRows, m_nCols);
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			result.m_data[row][col] = m_data[row][col] * instance.m_data[row][col];
		}
	}
	return result;
}

Matrix Matrix::operator*(float val) const
{
	Matrix result(m_nRows, m_nCols);
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			result.m_data[row][col] = m_data[row][col] * val;
		}
	}
	return result;
}

Matrix Matrix::operator/(Matrix& instance) const
{
	Matrix result(m_nRows, m_nCols);
	for (int row = 0; row < m_nRows; ++row)
	{
		for (int col = 0; col < m_nCols; ++col)
		{
			result.m_data[row][col] = m_data[row][col] / instance.m_data[row][col];
		}
	}
	return result;
}

Matrix Matrix::dot(Matrix& instance) const
{
	Matrix result(m_nRows, instance.m_nCols);
	for (int row = 0; row < m_nRows; row++)
	{
		for (int col = 0; col < instance.m_nCols; col++)
		{
			for (int inner = 0; inner < m_nCols; inner++)
			{
				result.m_data[row][col] += m_data[row][inner] * instance.m_data[inner][col];
			}
		}
	}
	return result;
}

unsigned int Matrix::get_numCols() const
{
	return m_nCols;
}

unsigned int Matrix::get_numRows() const
{
	return m_nRows;
}