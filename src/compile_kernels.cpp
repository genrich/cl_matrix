#include <iostream>
#include <fstream>

#include "ClService.hpp"

using namespace std;

extern ClService clSrvc;

int main ()
{
    cout << "Compiling kernels...\n";

    ifstream kernelsFile {"src/kernels.cl"};
    string kernelsString {istreambuf_iterator<char> {kernelsFile}, istreambuf_iterator <char> {}};

    cl::Program program {clSrvc.ctx, kernelsString};
    try
    {
        program.build (clSrvc.devices);

        auto byteSize = program.getInfo<CL_PROGRAM_BINARY_SIZES> () [0];
        auto binaries = program.getInfo<CL_PROGRAM_BINARIES> () [0];

        ofstream kernels {"kernels", ios::binary};
        kernels.write (binaries, byteSize);
        kernels.close ();

        cout << "done\n";
        return 0;
    }
    catch (cl::Error&)
    {
        cout << "Error:\n";
        cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG> (clSrvc.device) << endl;
    }
    return -1;
}
