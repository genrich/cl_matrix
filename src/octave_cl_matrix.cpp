#include <octave/config.h>
#include <octave/ops.h>
#include <octave/ov-re-mat.h>

#include "octave_cl_matrix.hpp"
#include "ClService.hpp"

using namespace std;

DEFINE_OCTAVE_ALLOCATOR (octave_cl_matrix);
DEFINE_OV_TYPEID_FUNCTIONS_AND_DATA (octave_cl_matrix, "cl_matrix", "cl_matrix");

static bool type_loaded = false;

extern ClService clSrvc;

void report_error (string msg)
{
    (*current_liboctave_error_handler) (msg.c_str ());
}

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
    os << "cl_matrix (" << matrix.rows << ", " << matrix.cols << ");  content: double (CL_MATRIX)\n";
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
    return matrix.byteSize ();
}

Matrix octave_cl_matrix::matrix_value (bool = false) const
{
  Matrix retval {matrix.rows, matrix.cols};
  matrix.copyTo (retval.fortran_vec());
  return retval;
}

static octave_cl_matrix* mul (const ClMatrix& mat1, const ClMatrix& mat2)
{
    try
    {
        return new octave_cl_matrix {mat1 * mat2};
    }
    catch (exception& e)
    {
        report_error (e.what ());
        return nullptr;
    }
}

static octave_cl_matrix* el_mul (const ClMatrix& mat1, const ClMatrix& mat2)
{
    try
    {
        return new octave_cl_matrix {mat1.el_mul (mat2)};
    }
    catch (exception& e)
    {
        report_error (e.what ());
        return nullptr;
    }
}

DEFBINOP_FN (mul,    cl_matrix, cl_matrix, mul)
DEFBINOP_FN (el_mul, cl_matrix, cl_matrix, el_mul)

DEFCONV(cl_matrix_to_matrix, octave_cl_matrix, octave_matrix)
{
  CAST_CONV_ARG (const octave_cl_matrix&);
  return new octave_matrix {v.matrix_value()};
}

DEFUN_DLD (cl_matrix, args, nargout,
"-*- texinfo -*-                                                           \n\
@deftypefn{Loadable Function} {@var{ret} =} cl_matrix (@var{double_matrix})\n\
Create OpenCL matrix                                                       \n\
@end deftypefn                                                             \n")
{
    int nargin = args.length ();
    if (nargin == 1 && (args (0).is_real_scalar () || args (0).is_real_matrix ()))
    {
        if (!type_loaded)
        {
            if (!clSrvc.initialized)
            {
                report_error ("Unable to initialize! " + clSrvc.statusMsg);
                return {};
            }

            octave_cl_matrix::register_type ();
            mlock ();

            INSTALL_BINOP (op_mul,    octave_cl_matrix, octave_cl_matrix, mul);
            INSTALL_BINOP (op_el_mul, octave_cl_matrix, octave_cl_matrix, el_mul);

            INSTALL_CONVOP (octave_cl_matrix, octave_matrix, cl_matrix_to_matrix);

            type_loaded = true;
        }


        Matrix m = args(0).matrix_value ();
        if (!error_state)
        {
            try
            {
                return octave_value {new octave_cl_matrix {m}};
            }
            catch (exception& e)
            {
                report_error (e.what ());
            }
        }
    }
    else
    {
        print_usage ();
    }
    return {};
}

DEFUN_DLD(sigmoid, args, nargout,
"-*- texinfo -*-                                                           \n\
@deftypefn {Mapping Function} {@var{cl_matrix} =} sigmoid (@var{cl_matrix})\n\
Apply @code{sigmoid} function to @var{cl_matrix}                           \n\
                                                                           \n\
@example                                                                   \n\
@group                                                                     \n\
   1                                                                       \n\
-------                                                                    \n\
     -X                                                                    \n\
1 + e                                                                      \n\
@end group                                                                 \n\
@end example                                                               \n\
@end deftypefn                                                             \n")
{
    int nargin = args.length ();
    if (nargin == 1 && octave_cl_matrix::static_type_id () == args (0).type_id ())
    {
        try
        {
            const octave_cl_matrix& m = dynamic_cast<const octave_cl_matrix&> (args (0).get_rep ());
            return octave_value {new octave_cl_matrix {m.cl_matrix_value ().sigmoid ()}};
        }
        catch (exception& e)
        {
            report_error (e.what ());
        }
    }
    else
    {
        print_usage ();
    }
    return {};
}
