#include <string>

#include <clAmdBlas.h>

using namespace std;

#define BOOST_TEST_MODULE ClMatrixTest

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <ClMatrix.hpp>

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyScalarTest)
{
    double data1[] = {2},
           data2[] = {3},
           data[1],
           expectedResult[] = {6};
    ClMatrix mat1{1, 1, data1},
             mat2{1, 1, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (data);

    BOOST_CHECK_EQUAL (data[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyVectors1Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           data[1],
           expectedResult[] = {11};
    ClMatrix mat1{1, 2, data1},
             mat2{2, 1, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (data);

    BOOST_CHECK_EQUAL (data[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyVectors2Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           data[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1{2, 1, data1},
             mat2{1, 2, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (data);

    BOOST_CHECK_EQUAL_COLLECTIONS (&data[0], &data[3], &expectedResult[0], &expectedResult[3]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyMatrixTest)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           data[4],
           expectedResult[] = {23, 34, 31, 46};
    ClMatrix mat1{2, 2, data1},
             mat2{2, 2, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (data);

    BOOST_CHECK_EQUAL_COLLECTIONS (&data[0], &data[3], &expectedResult[0], &expectedResult[3]);
}

