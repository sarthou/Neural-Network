#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>

template<typename TData, typename TSize = size_t>
class Matrix
{
public:

	using value_type = TData;
	using size_type = TSize;

	Matrix(size_type row_count, size_type col_count) :
		m_size{row_count, col_count}
	{
		m_data.resize(row_count);
		for(auto& row : m_data)
			row.resize(col_count);
	}

	Matrix(size_type row_count, size_type col_count, const value_type* data) :
		m_size{row_count, col_count}
	{
		m_data.resize(row_count);
		for(auto& row : m_data)
			row.resize(col_count);

		for(size_type row = 0 ; row < row_count ; row++)
			for(size_type col = 0 ; col < col_count ; col++)
				m_data[row][col] = data[row*col_count + col];
	}

	Matrix(size_type row_count, size_type col_count, const std::vector<std::vector<value_type>>& data) :
		m_size{row_count, col_count}
	{
		m_data.resize(row_count);
		for(auto& row : m_data)
			row.resize(col_count);

		if(data.size() != m_data.size())
		{
			std::cerr << "[" << __FILE__ << "@" << __LINE__ << "] "
					  << "Matrix::Matrix(): inconsistent number of rows in given vector" << std::endl;
			return;
		}
		for(size_type row = 0 ; row < row_count ; row++)
		{
			if(data[row].size() != m_data[row].size())
			{
				std::cerr << "[" << __FILE__ << "@" << __LINE__ << "] "
						  << "Matrix::Matrix(): inconsistent number of columns in given vector" << std::endl;
				return;
			}
			for(size_type col = 0 ; col < col_count ; col++)
			{
				m_data[row][col] = data[row][col];

			}
		}
	}

	value_type& operator()(size_type row, size_type col)
	{
		return m_data[row][col];
	}

	const value_type& operator()(size_type row, size_type col) const
	{
		return m_data[row][col];
	}

	std::vector<value_type>& get_row(size_type row)
	{
		return m_data[row];
	}

	size_type get_row_count() const
	{
		return m_size.row;
	}

	size_type get_col_count() const
	{
		return m_size.col;
	}

private:

	value_type& at(size_type row, size_type col)
	{
		if(row >= m_data.size() || col >= m_data[row].size())
			throw std::out_of_range("");
		return m_data[row][col];
	}

	const value_type& at(size_type row, size_type col) const
	{
		if(row >= m_data.size() || col >= m_data[row].size())
			throw std::out_of_range("");
		return m_data[row][col];
	}

	struct Size
	{
		size_type row;
		size_type col;
	};

	Size get_size() const
	{
		return m_size;
	}

	const Size m_size;

	std::vector<std::vector<value_type>> m_data;
};



#endif /* MATRIX_H_ */
