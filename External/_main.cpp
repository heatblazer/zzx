#include <iostream>
#include <cstdint>
#include "imatrix.h"

using namespace std;
using namespace mat;


union ptraddr {
    void* ptr;
    uint64_t val;
    uint8_t data[sizeof(uint64_t)];
};

static int random(int min, int max) //range : [min, max]
{
    static bool first = true;
    if (first)
    {
        srand( time(NULL) ); //seeding for the first time only!
        first = false;
    }
    return min + rand() % (( max + 1 ) - min);
}

int main_func()
{
    cout << "create a matrix of invalid type" << endl;
    try {
        imatrix<unsigned int, 10, 10> err;
    } catch (std::exception& e){
        cout << "exception - invalid matrix creation " << e.what() << endl;
    }

    cout << "create matrix" << endl;
    imatrix<int, 10, 10, eType::RowMinor> m;
    for(int i=0; i < 10; i++) {
        for(int j=0; j < 10; j++)
            m.setAt(i, j, random(1, 5));
    }
    m.print();
    cout << "get 2nd row" << endl;
    auto r1 = m.getRow(2);
    auto c1 = m.getCol(3);
    for(int i=0; i < 10; i++)
        cout << "[" << r1[i] << "]";
    cout << "\r\n-----" << endl;
    for(int i=0; i < 10; i++)
        cout << "[" << c1[i] << "]";

    cout << "\r\n-----" << endl;
    m.print();

    cout << "create transpose matrix " << endl;
    auto n = m.transpose();
    n.print();
    cout << "cofactor matrix m" << endl;
    m.cofactor();
    m.print();

    cout << "custom creation of a matrix row minor" << endl;

    imatrix<int, 2, 3> twod ;
    twod.setAt(0,0, 1);
    twod.setAt(0, 1, 2);
    twod.setAt(0, 2, 3);
    twod.setAt(1, 0, 4);
    twod.setAt(1, 1, 5);
    twod.setAt(1, 2, 6);
    twod.print();

    cout << "custom creation of a matrix row major" << endl;
    imatrix<int, 3, 2> twod2;
    twod2.setAt(0, 0, 1);
    twod2.setAt(1, 0, 2);
    twod2.setAt(2, 0, 3);
    twod2.setAt(0, 1, 4);
    twod2.setAt(1, 1, 5);
    twod2.setAt(2, 1, 6);
    twod2.print();

    cout << "using it as vector test " <<endl;

    imatrix<int, 1, 100> vec1;

    for(int j=0; j < 100; j++) {
        vec1.setAt(0, j, j * 10);
    }

    vec1.print();

    cout << "get row from a matrix " << endl;
    imatrix<int, 6, 6, eType::RowMajor> m1;
    for(int i=0; i < 6; i++) {
        for(int j=0; j < 6; j++)
            m1.setAt(j, i, j * 6 + i);
    }
    m1.print();
    cout << "5 th row " << endl;
    auto row5 = m1.getRow(5);
    try {
        auto row10 = m1.getRow(10); //exception will trigger
    } catch (std::exception& e) {
        cout << e.what() << endl;
    }

    puts("..");
    for(size_t i=0; i < m1.rows(); i++) {
        cout << "[" << row5[i] << "]";
    }


    puts("..");
    cout << "rotate left 3 times " << endl;
//    m1.rotateLeft(3);
    m1.print();
//TODO - finish the LU DECOMPOSE
    {
        cout << "LU decompose " << endl ;
        imatrix<int, 4, 4> lu;
        //row 0
        lu.setAt(0, 0, 2);
        lu.setAt(0, 1, 4);
        lu.setAt(0, 2, 3);
        lu.setAt(0, 3, 5);

        //row 1
        lu.setAt(1, 0, -4);
        lu.setAt(1, 1, -7);
        lu.setAt(1, 2, -5);
        lu.setAt(1, 3, -8);

        //row 2
        lu.setAt(2, 0, 6);
        lu.setAt(2, 1, 8);
        lu.setAt(2, 2, 2);
        lu.setAt(2, 3, 9);

        //row 3
        lu.setAt(3, 0, 4);
        lu.setAt(3, 1, 9);
        lu.setAt(3, 2, -2);
        lu.setAt(3, 3, 14);

        auto lu2 = lu.transpose();
        lu.print();
        puts("--");
        lu2.print();
        imatrix<int, 4,4> l;
        imatrix<int, 4,4> u;

        lu.LUDecompose(l, u);
        cout << "L " << endl;
        l.print();
        cout << "U " << endl;
        u.print();
        cout << "end LU test" << endl;
    }

    {
        cout << "set n row with elements 1 2 3 4 5" << endl;
        imatrix<int, 5, 5> m;
        std::array<int, 5> a {1,2,3,4,5};
        m.setRow(4,a);
        m.setCol(1, a);
        m.print();
        cout << "end set row test";
    }
    cout << "-- end test --" << endl;

    return 0;
}
