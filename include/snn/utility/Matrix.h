#ifndef UTIL_MATRIX_H_
#define UTIL_MATRIX_H_

#include <vector>
#include <cstring>
#include <iostream>

template<typename TData, typename TSize = size_t>
class Matrix
{
public:

	using value_type = TData;
	using size_type = TSize;

	Matrix(size_type row_count, size_type col_count) :
		m_row_count(row_count), m_col_count(col_count), m_data{ new value_type[row_count*col_count] }, row_index{ new unsigned int[row_count]}
	{
		for (unsigned int i = 0; i < row_count; i++)
			row_index[i] = i*col_count;
	}

	Matrix(size_type row_count, size_type col_count, const value_type* data) :
		m_row_count(row_count), m_col_count(col_count), m_data{ new value_type[row_count*col_count] }, row_index{ new unsigned int[row_count] }
	{
		std::memcpy(m_data, data, row_count*col_count);
		for (unsigned int i = 0; i < row_count; i++)
			row_index[i] = i*col_count;
	}

	Matrix(size_type row_count, size_type col_count, const std::vector<std::vector<value_type>>& data) :
		m_row_count(row_count), m_col_count(col_count), m_data{ new value_type[row_count*col_count] }, row_index{ new unsigned int[row_count] }
	{
		if (data.size() != row_count)
		{
			std::cerr << "[" << __FILE__ << "@" << __LINE__ << "] "
				<< "Matrix::Matrix(): inconsistent number of rows in given vector" << std::endl;
			return;
		}
		for (size_type row = 0; row < row_count; row++)
		{
			if (data[row].size() != col_count)
			{
				std::cerr << "[" << __FILE__ << "@" << __LINE__ << "] "
					<< "Matrix::Matrix(): inconsistent number of columns in given vector" << std::endl;
				return;
			}
			for (size_type col = 0; col < col_count; col++)
			{
				m_data[row*col_count + col] = data[row][col];

			}
		}
		for (unsigned int i = 0; i < row_count; i++)
			row_index[i] = i*col_count;
	}

	Matrix(const Matrix&) = delete;
	Matrix& operator=(const Matrix&) = delete;

	~Matrix()
	{
		delete[] m_data;
		delete[] row_index;
	}

	inline value_type& operator()(size_type row, size_type col)
	{
		return m_data[row_index[row] + col];
	}

	inline const value_type& operator()(size_type row, size_type col) const
	{
		return m_data[row_index[row] + col];
	}

	inline value_type& operator[](size_type index)
	{
		return m_data[index];
	}

	inline value_type* get_row(size_type row)
	{
		return &m_data[row_index[row]];
	}

	inline size_type get_row_count() const
	{
		return m_row_count;
	}

	inline size_type get_col_count() const
	{
		return m_col_count;
	}

	value_type* data()
	{
		return m_data;
	}

	const value_type* data() const
	{
		return m_data;
	}

private:

	const size_type m_row_count;
	const size_type m_col_count;

	value_type* m_data;
	unsigned int* row_index;
};

#endif /* UTIL_MATRIX_H_ */
