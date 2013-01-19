#include <string>

using namespace std;

#define BOOST_TEST_MODULE ClMatrixTest

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <ClMatrix.hpp>

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyScalarTest)
{
    double data1[] = {2},
           data2[] = {3},
           result[1],
           expectedResult[] = {6};
    ClMatrix mat1 {1, 1, data1},
             mat2 {1, 1, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyVectors1Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[1],
           expectedResult[] = {11};
    ClMatrix mat1 {1, 2, data1},
             mat2 {2, 1, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyVectors2Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {2, 1, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[3], &expectedResult[0], &expectedResult[3]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyMatrixTest)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {23, 34, 31, 46};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1 * mat2;
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[3], &expectedResult[0], &expectedResult[3]);
}

BOOST_AUTO_TEST_CASE (ClMatrixMultiplyMatrixErrorTest)
{
    double data1[] = {1, 2},
           data2[] = {3, 4};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    BOOST_CHECK_THROW (mat1 * mat2, runtime_error);
}

BOOST_AUTO_TEST_CASE (ClMatrixElMultiplyTest)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {5, 12, 21, 32};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.el_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[3], &expectedResult[0], &expectedResult[3]);
}

BOOST_AUTO_TEST_CASE (ClMatrixElMultiplyErrorTest)
{
    double data1[] = {1, 2},
           data2[] = {3};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 1, data2};

    BOOST_CHECK_THROW (mat1.el_mul (mat2), runtime_error);
}

BOOST_AUTO_TEST_CASE (ClMatrixSigmoidTest)
{
    constexpr int size = 4;
    double data[] = {1, 2, 3, 4},
           result[size];
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.sigmoid ();
    mat.copyTo (result);

    constexpr double tolerance = 1e-10;
    for (int i = 0; i < size; i++)
    {
        BOOST_CHECK_SMALL (result[i] - (1 / (1 + exp (-data[i]))), tolerance);
    }
}
