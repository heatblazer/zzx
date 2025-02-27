#pragma once
#include <cstring>
#include <cstdio>
#include <functional>
#include <iostream>
#include <array>
///
/// \brief compile time only matrix class
///

template <typename T> struct isvalid { static constexpr bool value = false ;};

template <> struct isvalid<int> { static constexpr bool value = true; };

template <> struct isvalid<short> { static constexpr bool value = true; };

template <> struct isvalid<float> { static constexpr bool value = true; };

template <> struct isvalid<double> { static constexpr bool value = true; };

template <> struct isvalid<long long> { static constexpr bool value = true; };

template <> struct isvalid<char> { static constexpr bool value = true; };

namespace mat {

enum class eType {
    RowMajor,
    RowMinor,
    Unknown,
    Size = Unknown
};

template<typename T=int, size_t ROW=3, size_t COL=3, eType type = eType::RowMajor>
class imatrix final
{
    struct {
        T data[ROW * COL];
        T temp[ROW * COL]; //helper
        T cof[ROW * COL]; // + - + - ...
    } m;
//    eType type;

    inline void setat(T val, int x, int y, T(d[ROW * COL]))
    {
        if constexpr (type == eType::RowMajor)
            d[y * ROW + x] = val;
        if constexpr (type == eType::RowMinor)
            d[x * ROW + y] = val;

    }

    inline T getat(int x, int y, const T(d[ROW * COL]))
    {
        if constexpr (type == eType::RowMajor)
            return d[y * ROW + x];
        if constexpr (type == eType::RowMinor)
            return d[x * ROW + y];

    }

    void mult_vec_by(T v, std::array<T, ROW>& d)
    {
        for(size_t i=0; i < d.size(); i++) {
            d[i] *= v;
        }
    }

    inline T getDiff(T x, T y)
    {
        return x / y;
    }

    ///
    /// \brief foreach row/col in matrix perform a fucntion
    /// \param fn - function for action per item
    /// thorw logical exception
    void loop(std::function<void(int, int)> fn)
    {
        if (type == eType::RowMajor) {
            for(size_t i=0; i < COL; i++) {
                for(size_t j=0; j < ROW; j++) {
                    fn(i, j);
                }
            }
        } else if (type == eType::RowMinor) {
            for(size_t i=0; i < COL; i++) {
                for(size_t j=0; j < ROW; j++) {
                    fn(j, i);
                }
            }
        } else {
            throw std::logic_error("Unknown matrix type loop!");
        }
    }

    T det2x2()
    {
        auto r1 = getRow(0);
        auto r2 = getRow(1);
        return r1[0] * r2[1] - r1[1] * r2[0];
    }

    T det3x3()
    {
        auto r1 = getRow(0);
        auto r2 = getRow(1);
        auto r3 = getRow(2);
        return
            (r1[0] * ((r2[1] * r3[2]) - (r2[2] * r3[1]))) -
            (r1[1] * ((r2[2] * r3[0]) - (r3[2] * r2[0]))) +
            (r1[2] * ((r2[0] * r3[1]) - (r3[0] * r2[1])));
    }

public:
    explicit imatrix()
    {
        if constexpr (!isvalid<T>::value) {
            throw std::logic_error{"exception: not a valid type provided"};
        }
        int c = 1;
        memset(&m, 0, sizeof(m));

        switch (type) {
        case eType::RowMajor: {
            for(size_t i=0; i < COL; i++) {
                for (size_t j=0; j < ROW; j++) {
                    setat(c, i, j, m.cof);
                    c *= -1;
                }
            }
            break;
        }
        case eType::RowMinor: {
            for(size_t i=0; i < COL; i++) {
                for (size_t j=0; j < ROW; j++) {
                    setat(c, j, i, m.cof);
                    c *= -1;
                }
            }
            break;
        }
        default:
            break;
        }
    }

    void setRow(int col, const std::array<T, ROW>& data)
    {
        loop([this, col, data](int i, int j)
        {
            setAt(col, j, data[j]);
        });
    }

    void setCol(int row, const std::array<T, ROW>& data)
    {
        loop([this, row, data](int i, int j)
             {
                 setAt(i, row, data[i]);
             });
    }

    ///
    /// \brief getRow - return a row of a matrix given a column num
    /// \param col
    /// \return row at col num
    ///
    std::array<T, ROW> getRow(int col)
    {

        std::array<T, ROW> row;
        loop([this, &col, &row](int i, int j)
             {
                 row[j] = getAt(col, j);
             });
        return row ;
    }

    ///
    /// \brief getCol
    /// \param row
    /// \return column at row
    ///
    std::array<T, COL> getCol(int row)
    {
        std::array<T, COL> col;
        loop([this, &col, &row](int i, int j)
        {
            col[i] = getAt(i, row);
        });
        return col ;
    }

    ///
    /// \brief getPrimaryDiag
    /// \param
    /// \return primary diagonal
    ///
    std::array<T, COL> getPrimaryDiag()
    {
        std::array<T, COL> diag;

        if constexpr (ROW != COL) {
            throw std::logic_error{"can't take diag of non square matrix"};
        }

        loop([this, &diag](int i, int j)
        {
            diag[i] = getAt(i,i);
        });
        return diag ;
    }

    ///
    /// \brief getAt - get item at row/col
    /// \param x
    /// \param y
    /// \return
    ///
    inline T getAt(int i, int j)
    {
        return getat(i, j, m.data);
    }

    ///
    /// \brief setAt - set item at row/col
    /// \param x
    /// \param y
    /// \param val
    ///
    inline void setAt(int i, int j, const T& val)
    {
        setat(val, i, j, m.data);
    }

    ///
    /// \brief rotateLeft - rotates N times all rows
    /// \param n
    ///
    constexpr void rotateLeft(unsigned int n)
    {
        loop([this, n](int i, int j)
        {
            m.temp[j * COL + i] = getat(i, (j + n) % COL, m.data);
        });

    }

    /// TODO
    /// \brief det
    /// \return determinant
    ///
    constexpr T det()
    {
        if constexpr (ROW != COL) {
            throw std::logic_error{"det can be calculated only on square matrices"};
        }
        if (ROW == 2) return det2x2();
        if (ROW == 3) return det3x3();
        T d=1;
        loop([this, &d](int i, int j)
        {
            d *= getAt(i, i);
        });
        return d;
    }

    ///
    /// \brief cofactor
    /// sets coefficients by the cofactor fashion:
    /// [1,-1,1,-1,1,...]
    /// [1,-1,1,-1,1,...]
    /// ...
    /// [1,-1,1,-1,1,...]
    constexpr void cofactor()
    {
        loop([this](int i, int j)
        {
            setat(getat(i,j, m.cof) * getat(i,j, m.data), i, j, m.data);
        });
    }

    ///
    /// \brief transpose a matrix
    /// [1,2,3]     [1,4,7]
    /// [4,5,6] --> [2,5,8]
    /// [7,8,9]     [3,6,9]
    /// \return
    ///
    constexpr imatrix<T, ROW, COL, type>  transpose()
    {
        imatrix<T, ROW, COL, type> newmat;

        for(size_t i=0; i < ROW; i++) {
            for(size_t j=0; j < COL; j++)
            {
                newmat.setat(getat(j, i, m.data), i, j, newmat.m.data);
                newmat.setat(getat(j, i, m.cof), i, j, newmat.m.cof);
            }
        }
        return newmat;
    }

// operator sections for dot/cross add, sub, etc.
    void operator *(const T& scalar)
    {
        loop([this, scalar](int i, int j)
        {
            setat(getat(i, j, m.data) * scalar, i, j, m.data);
        });
    }

    ///
    /// \brief print print matrix
    ///
    void print()
    {
        if (COL == 1) {
            for(size_t i=0; i < ROW; i++) {
                printf("[%d]", m.data[i]);
            }
            puts("");
        } else if(ROW == 1) {
            for(size_t i=0; i < COL; i++) {
                printf("[%d]", m.data[i]);
            }
            puts("");
        } else {
            if (type == eType::RowMajor) {
                for(size_t i=0; i < COL; i++) {
                    for(size_t j=0; j < ROW; j++) {
                        printf("[%d]", getAt(i, j));
                    }
                    puts("");
                }
            } else if (type == eType::RowMinor) {
                for(int i=0; i < COL; i++) {
                    for(int j=0; j < ROW; j++) {
                        printf("[%d]", getAt(j, i));
                    }
                    puts("");
                }
            }
        }
    }

    inline constexpr size_t rows() const { return ROW; }

    inline constexpr size_t cols() const { return COL; }


    ///
    /// \brief LUDecompose - using gaussian elimination to create 2 triangular
    /// matrices
    /// \param L - lower triangle
    /// \param U - upper triangle
    ///
    void LUDecompose(imatrix<T, ROW, COL, type>& L, imatrix<T, ROW, COL, type>& U)
    {
        for(size_t i=0; i < COL; i++) {
            for(size_t j=0; j < ROW; j++) {
                L.setAt(i,i, 1);
            }
        }//set identity for L
        for(size_t i=0; i < COL; i++)
            U.setRow(i, getRow(i));

        auto diag = U.getPrimaryDiag();
        for(size_t i=0; i < diag.size(); i++)
        {
            for(size_t j=1+i; j < COL; j++)
            {
                std::array<T, ROW> temp = U.getRow(i);
                auto r = U.getRow(j);
                auto diff = getDiff(r[i], diag[i]);
                for(size_t h=0; h < ROW; h++) temp[h] *= diff; //mult first temp row
                for(size_t h=0; h < ROW; h++) r[h] -= temp[h];
                L.setAt(j,i, diff);
                U.setRow(j, r);
            }
            diag = U.getPrimaryDiag();
        }
    }
};

} //!mat

