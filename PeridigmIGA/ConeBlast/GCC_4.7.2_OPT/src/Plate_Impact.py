import mesh_functions as ms
import stretching_functions as sf
from igakit.cad import *
import numpy as np
# Specify origin
O=np.array([0,0,0])

# Lines for construction
C3=line(p0=(0.00,0.00,0), p1=(1.8,0.00,0))
C4=line(p0=(0.00,1.8,0), p1=(1.8,1.80,0))

# Call geometry generation functions for the background with optional argument n which indicates elevation of background elements
# n=1 for background quadratic, 2 for cubic
S=ms.generate_unif_background(C3, C4, 1.8, np.array([150,150,150]), n=1)

#print(S.knots)

#S.refine(0, 0.45)


#for i in range(S.control.shape[0]-1):
 #   for j in range(S.control.shape[1]-1):
 #      print(S.control[i, j, :])
#print(S.control)
        
#### EXPERIMENTAL : SETTING ALL ASSOCIATED VOLUMES IDENTICALLY : QUADRATURE AND CORRECTION IS TOTALLY CONSISTENT IN THIS SCENARIO #########
xt1=0.045-0.025-0.025/200 #spacing PLUS the padding (half the width of the associated volume)
xt2=0.045+0.01+0.025/200
yt1=0.05-0.025/2
yt2=0.05+0.01-0.025/2
# Call foreground geometry generation with discritization and geometry parameters
G1=ms.generate_unif_foreground(O, np.array([0.025, 0.025, 0.0]), np.array([100,100,1]))
G1=sf.translate(G1, xt1, yt1, 0)
G2=ms.generate_unif_foreground(O, np.array([0.025, 0.025, 0.0]), np.array([100,100,1]))
G2=sf.translate(G2, xt2, yt2, 0)
G=sf.fg_superpose(G1,G2)
#Biaxial stretching
#G=sf.lin_stretch(0, 0.002, 1.0, G)
#G=sf.lin_stretch(1, 0.004, 1.0, G)

#S=sf.nonlinear_parameterization(S) # Here the function nonlinear parameterization is called

# Save geometry for visualization, and background as .dat files. 
ms.save_geometry(G, S, 1)
ms.vis_background()
