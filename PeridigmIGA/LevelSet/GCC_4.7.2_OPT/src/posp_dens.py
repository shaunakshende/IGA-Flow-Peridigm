from igakit.io import PetIGA,VTK
from numpy import linspace
import glob

# read in discretization info and potentially geometry
nrb = PetIGA().read("igaF.dat")


# write a function to sample the nrbs object (100 points from beginning to end)
uniform = lambda U: linspace(U[0], U[-1], 150)

for infile in glob.glob("velS*.dat"):
# read in solution vector as a numpy array
#	igafile = "pfc_iga.dat"
#	nrb = PetIGA().read("pfc_iga.dat")
	sol = PetIGA().read_vec(infile,nrb)
	outfile = infile.split(".")[0] + ".vtk"
# write a binary VTK file
	VTK().write(outfile,       # output filename
            nrb,                    # igakit NURBS object
            fields=sol,             # sol is the numpy array to plot
#            sampler=uniform,        # specify the function to sample points
                        scalars={'Density':0, 'Temperature':4},
                        vectors={'vel':[1,2,3]}
)
