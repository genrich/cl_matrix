#include <string>

using namespace std;

#define BOOST_TEST_MODULE ClMatrixTest

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <ClMatrix.hpp>

BOOST_AUTO_TEST_CASE (mul_SingleElementMitrixTest)
{
    double data1[] = {2},
           data2[] = {3},
           result[1],
           expectedResult[] = {6};
    ClMatrix mat1 {1, 1, data1},
             mat2 {1, 1, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (mul_Vectors1Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[1],
           expectedResult[] = {11};
    ClMatrix mat1 {1, 2, data1},
             mat2 {2, 1, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (mul_Vectors2Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {2, 1, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (mul_Test)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {23, 34, 31, 46};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (mul_ErrorTest)
{
    double data1[] = {1, 2},
           data2[] = {3, 4};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    BOOST_CHECK_THROW (mat1.mul (mat2), runtime_error);
}

BOOST_AUTO_TEST_CASE (trans_mul_Test)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {17, 39, 23, 53};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.trans_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (trans_mul_VectorsTest)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1.trans_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (mul_trans_Test)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {26, 38, 30, 44};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.mul_trans (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (mul_trans_Vectors1Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[1],
           expectedResult[] = {11};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 2, data2};

    ClMatrix mat = mat1.mul_trans (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL (result[0], expectedResult[0]);
}

BOOST_AUTO_TEST_CASE (mul_trans_Vectors2Test)
{
    double data1[] = {1, 2},
           data2[] = {3, 4},
           result[4],
           expectedResult[] = {3, 6, 4, 8};
    ClMatrix mat1 {2, 1, data1},
             mat2 {2, 1, data2};

    ClMatrix mat = mat1.mul_trans (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (add_Test)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 2, INFINITY, -2},
           result[4],
           expectedResult[] = {6, 4, INFINITY, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.add (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (add_ErrorTest)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {1, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {1, 2, data2};

    BOOST_CHECK_THROW (mat1.add (mat2), runtime_error);
}

BOOST_AUTO_TEST_CASE (sub_Test)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 2, INFINITY, 2},
           result[4],
           expectedResult[] = {-4, 0, -INFINITY, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.sub (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (sub_ErrorTest)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {1, 2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 1, data2};

    BOOST_CHECK_THROW (mat1.sub (mat2), runtime_error);
}

BOOST_AUTO_TEST_CASE (sub_scalar_Test)
{
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {-4, -3, -2, -1};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.sub (5);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (scalar_sub_Test)
{
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {4, 3, 2, 1};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.subtrahend (5);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (div_scalar_Test)
{
    double data[] = {2, 4, 8, 16},
           result[4],
           expectedResult[] = {1, 2, 4, 8};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.div (2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (div_scalar0_Test)
{
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {INFINITY, INFINITY, INFINITY, INFINITY};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.div (0);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (scalar_div_Test)
{
    double data[] = {2, 4, 8, 0},
           result[4],
           expectedResult[] = {4, 2, 1, INFINITY};
    ClMatrix mat1 {2, 2, data};

    ClMatrix mat = mat1.divisor (8);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (el_mul_Test)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {5, 6, 7, 8},
           result[4],
           expectedResult[] = {5, 12, 21, 32};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.el_mul (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (el_mul_ErrorTest)
{
    double data1[] = {1, 2},
           data2[] = {3};
    ClMatrix mat1 {1, 2, data1},
             mat2 {1, 1, data2};

    BOOST_CHECK_THROW (mat1.el_mul (mat2), runtime_error);
}

BOOST_AUTO_TEST_CASE (el_div_Test)
{
    double data1[] = {2, 4, 0, 16},
           data2[] = {2, 0, 4, -8},
           result[4],
           expectedResult[] = {1, INFINITY, 0, -2};
    ClMatrix mat1 {2, 2, data1},
             mat2 {2, 2, data2};

    ClMatrix mat = mat1.el_div (mat2);
    mat.copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (el_div_ErrorTest)
{
    double data1[] = {1, 2, 3, 4},
           data2[] = {1};
    ClMatrix mat1 {2, 2, data1},
             mat2 {1, 1, data2};

    BOOST_CHECK_THROW (mat1.el_div (mat2), runtime_error);
}

BOOST_AUTO_TEST_CASE (uminus_Test)
{
    double data[] = {2, INFINITY, 0, 16},
           result[4],
           expectedResult[] = {-2, -INFINITY, 0, -16};
    ClMatrix mat {2, 2, data};

    mat.uminus ().copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (transpose_Test)
{
    double data[] = {1, 2, 3, 4},
           result[4],
           expectedResult[] = {1, 3, 2, 4};
    ClMatrix mat {2, 2, data};

    mat.transpose ().copyTo (result);

    BOOST_CHECK_EQUAL_COLLECTIONS (&result[0], &result[4], &expectedResult[0], &expectedResult[4]);
}

BOOST_AUTO_TEST_CASE (sigmoid_Test)
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

BOOST_AUTO_TEST_CASE (eventTest)
{
    double data[] {1, 2, 3, 4},
           result[4];
    ClMatrix mat {2, 2, data};

    auto m = mat.add (1);
    m.copyTo (result);
    m.copyTo (result);

    BOOST_CHECK (true);
}
