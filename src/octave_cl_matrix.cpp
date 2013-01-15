#include <octave/config.h>
#include <octave/ops.h>
#include <octave/ov-re-mat.h>

#include "octave_cl_matrix.hpp"
#include "ClAmdBlasService.hpp"

using namespace std;

DEFINE_OCTAVE_ALLOCATOR (octave_cl_matrix);
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_cl_matrix, "cl_matrix", "cl_matrix");

static bool type_loaded = false;

extern ClAmdBlasService clSrvc;

octave_cl_matrix::octave_cl_matrix()
    :matrix (1, 1)
{
}

octave_cl_matrix::octave_cl_matrix (const Matrix& m)
    :matrix (m.rows(), m.cols(), m.data())
{
}

octave_cl_matrix::octave_cl_matrix (ClMatrix m)
    :matrix (move (m))
{
}

octave_cl_matrix::~octave_cl_matrix()
{
}

const ClMatrix& octave_cl_matrix::cl_matrix_value() const
{
    return matrix;
}

void octave_cl_matrix::print (std::ostream& os, bool pr_as_read_syntax) const
{
    os << "cl_matrix(" << matrix.rows << ", " << matrix.cols << "); view content: double(cl_matrix_value)\n";
}

bool octave_cl_matrix::is_constant () const
{
    return true;
}

bool octave_cl_matrix::is_defined () const
{
    return true;
}

dim_vector octave_cl_matrix::dims () const
{
    return dim_vector (matrix.rows, matrix.cols);
}

size_t octave_cl_matrix::byte_size () const
{
    return matrix.cols * matrix.rows * sizeof(double);
}

Matrix octave_cl_matrix::matrix_value (bool = false) const
{
  Matrix retval {matrix.rows, matrix.cols};
  matrix.copyTo (retval.fortran_vec());
  return retval;
}

static octave_cl_matrix* mul (const ClMatrix& mat1, const ClMatrix& mat2)
{
    return new octave_cl_matrix (mat1 * mat2);
}

DEFBINOP_FN (mul, cl_matrix, cl_matrix, mul)

DEFCONV(cl_matrix_to_matrix, octave_cl_matrix, octave_matrix)
{
  CAST_CONV_ARG (const octave_cl_matrix&);
  return new octave_matrix {v.matrix_value()};
}

DEFUN_DLD (cl_matrix, args, nargout,
"cl_matrix_value = cl_matrix(double_matrix)\n\
\nCreates a OpenCl matrix.")
{
    int nargin = args.length ();
    if (nargin == 1 && (args (0).is_real_scalar () || args (0).is_real_matrix ()))
    {
        if (!type_loaded)
        {
            if (!clSrvc.initialized)
            {
                octave_stdout << "Error: unable to initialize clAmdBlas!\n";
                return octave_value_list();
            }

            octave_cl_matrix::register_type ();
            // mlock ();

            INSTALL_BINOP (op_mul, octave_cl_matrix, octave_cl_matrix, mul);

            INSTALL_CONVOP (octave_cl_matrix, octave_matrix, cl_matrix_to_matrix);

            type_loaded = true;
        }


        Matrix m = args(0).matrix_value ();
        if (!error_state)
        {
            try
            {
                return octave_value (new octave_cl_matrix (m));
            }
            catch (std::exception& e)
            {
                octave_stdout << "Error: " << e.what() << "\n";
            }
        }
    }
    else
    {
        print_usage ();
    }
    return octave_value_list ();
}

DEFUN_DLD(sigmoid, args, nargout, "Sigmoid function applied to cl_matrix elements")
{
    int nargin = args.length ();
    if (nargin == 1 && octave_cl_matrix::static_type_id () == args (0).type_id ())
    {
        octave_stdout << "unimplemented\n";
    }
    else
    {
        print_usage ();
    }
    return octave_value_list ();
}
