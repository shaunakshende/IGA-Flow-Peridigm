
//Techos
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include "Teuchos_GlobalMPISession.hpp"

//Peridigm Headers
#include <Peridigm.hpp>
#include <Peridigm_TextFileDiscretization.hpp>
#include "Peridigm_HorizonManager.hpp"

//Epetra
#include <Epetra_ConfigDefs.h> 
#ifdef HAVE_MPI
  #include <Epetra_MpiComm.h>
#else
  #include <Epetra_SerialComm.h>
#endif
#include <Teuchos_RCP.hpp>


// Petsc/PetIGA header files
#include "petiga.h"
#include "../src/petigagrid.h"
#include <petscsys.h>
#include <../../../../../src/sys/fileio/mprint.h>
#include <petscblaslapack.h>
#include <petsc/private/tsimpl.h>


// Boost parallelization libraries
#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/distributed/distributed_graph_utility.hpp>
#include <boost/graph/distributed/local_subgraph.hpp>
#include <boost/graph/iteration_macros.hpp>


// Standard Library includes
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

//using namespace std;
using namespace Teuchos;
using namespace PeridigmNS;

#define SQ(A) ((A)*(A))
#define for1(i,n) for(int i=0; i<(n); i++)
#define for2(i,j,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++)
#define for3(i,j,k,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++)
#define for4(i,j,k,l,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++)
#define for5(i,j,k,l,m,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++)
#define for6(i,j,k,l,m,o,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++) for(int o=0; o<(n); o++)
#define for7(i,j,k,l,m,o,p,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++) for(int o=0; o<(n); o++) for(int p=0; p<(n); p++)
#define for8(i,j,k,l,m,o,p,q,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++) for(int o=0; o<(n); o++) for(int p=0; p<(n); p++) for(int q=0; q<(n); q++)

//// Data Structures for Immersed-IGA-PD FSI ////

//Information we need for the background discretization
//For example Lx,Ly,Lz are the lengths of the background domain in x,y,z
//Nx,Ny,Nz are the numbers of elements in each directions
//A1,Ap,An are the n+1,alpha,n levels of the acceleration
//V1,Vp,Vn are the n+1,alpha,n levels of the velocity
//dA is the increment of the acceleration. This is what we are solving for
//Alpha_m,Alpha_f,Gamma,Beta are parameters for the generalized alpha method
typedef struct {

  IGA iga;
  PetscReal Lx,Ly,Lz,mu,lamda,kappa,thickness,nen;
  Vec A1,Ap,An,V1,Vp,Vn,D1,Dp,Dn,Aa,Va,Da,V0,A0,D0,dA,density;
  PetscInt  Nx,Ny,Nz,max_its;
  PetscReal Cv,R,F[4],Cp,p0;

  /*PetscReal fluidForceX,fluidForceY;*/
  /*Vec fluidResidual, numSolidNodesInElement, solidVolumeInElement;*/

  PetscReal TimeRestart;
  //  PetscReal currentTime,timeStep;
  PetscInt  StepRestart;
  //  PetscInt  stepNumber;
  PetscInt  FreqRestarts;
  //  PetscInt  FreqResults;

  PetscReal Alpha_m,Alpha_f,Gamma,Beta;

  PetscReal totalCurrentExplosiveVolume;
  PetscReal totalInitialExplosiveVolume;
} AppCtx;

typedef struct
{

  //UNDEX quantities
  PetscReal nodalVolume; //volume associated with the particle. It is used as the integration weight in nodal integration.
  //For quadrature volume update, new variables:
  double nodalDensity;
  double nodalDensityInitial;
  double nodalPressure;
  double nodalVolumeInitial;
  //Setting up for multi-material: 1 for RDX, 0 for immersed solid
  //for now.
  double material;
  //UNDEX quantities

  PetscReal support[2]; //support of the particle
  PetscReal supportInverse[2];
  PetscReal searchRadius[2]; //neighbor search radius of the particle


  PetscInt  orderOfBasis; //Order of basis for the derivative operator
  PetscInt  initialNumberOfNeighbors;

  PetscReal velocityGradient[4]; //velocity gradient on each particle. 2x2 for 2D. dvx/dx,dvx/dy,dvy/dx,dvy/dy for indexing 0-->4

  PetscReal currentDeformationGradient[4];
  PetscReal alphaDeformationGradient[4];
  PetscReal DeformationGradientOld[4];

  PetscReal determinantCurrentDeformationGradient;
  PetscReal determinantalphaDeformationGradient;

  PetscReal internalForce[2];
  PetscReal inertia[2];
  PetscReal bodyForce[2];
  PetscReal residual[2];

  PetscReal alphaNodalVolume;
  PetscReal currentNodalVolume;
  PetscReal referenceNodalVolume;

  PetscReal alphaDensity;
  PetscReal currentDensity;
  PetscReal referenceDensity;

  PetscReal weightedVolume;

  PetscInt  ID; //ID of the particle.
  PetscReal initialCoord[2]; //Initial coordinates of the particle for 2D
  PetscReal referenceCoord[2]; //Reference coordinates of the particle for 2D
  PetscReal currentCoord[2]; //Current coordinates of the particle for 2D
  PetscReal hvect[2];//dx, dy, and dz for each particle for 2D
  PetscReal totalPhysicalDisplacement[2]; //Displacement of the particle between two consecutive time steps in the two directions
  PetscReal totalPhysicalDisplacementOldIteration[2]; //Displacement of the particle in the previous iteration
  PetscReal totalPhysicalDisplacementOldStep[2]; //Displacement of the particle in the previous time step
  PetscReal totalPhysicalVelocity[2]; //Velocity of the particle at the current time position
  PetscReal totalPhysicalVelocityOldIteration[2]; //Velocity of the particle at the end of the previous iteration
  PetscReal totalPhysicalVelocityOldStep[2]; //Velocity of the particle at the end of the previous time step
  PetscReal totalPhysicalAcceleration[2]; //Acceleration of the particle at the current time position
  PetscReal totalPhysicalAccelerationOldIteration[2]; //Acceleration of the particle at the end of the previous iteration
  PetscReal totalPhysicalAccelerationOldStep[2]; //Acceleration of the particle at the end of the previous time step
  PetscReal AccelerationIncrement[2]; //Acceleration increment of the particle
  PetscReal totalStress[3]; //stress of particle at current time. For 2D we have 4 components but due to symmetry it collapses to 3 components
                            //The numbering is 0->sigmaxx, 1->sigmayy, 2->sigmaxy
  PetscReal totalStress0[3];//stress of particle at the end of the time step (after iterations have been completed and after we have rotated fully for objectivity)
  PetscReal totalStrain[3];//strain of the particle at current time. Same numbering as above
  PetscReal EpsilonPlastEq;//Equivalent plastic strain. Basically a non decreasing quantity that accumulates the plastic strain in every time step.
  PetscInt  check;

  PetscInt  numNeighbors;
  PetscInt  *neighbors;
  PetscReal *shapeFunction[2];
  PetscReal *totalStressBondLevel[3];
  PetscReal *totalStress0BondLevel[3];
  PetscReal *EpsilonPlastEqBondLevel;
  PetscReal *velocityGradientBondLevel[4];
  PetscReal *currentDeformationGradientBondLevel[4];
  PetscReal *alphaDeformationGradientBondLevel[4];
  PetscReal *DeformationGradientOldBondLevel[4];
  PetscReal *determinantCurrentDeformationGradientBondLevel;
  PetscReal *determinantalphaDeformationGradientBondLevel;

  PetscReal H;
  PetscReal Hold;

  PetscReal *HBondLevel;
  PetscReal *HoldBondLevel;

  PetscScalar *N0;
  PetscScalar *Nx;
  PetscScalar *Ny;
  PetscScalar *Nz;
  PetscInt *map;

} POINTS;

typedef struct
{

  char integrationMethod[256]; //method to compute the gradient shape functions. Base, RK, GMLS, etc
  PetscInt  kernelType;
  PetscInt  orderOfBasis;
  PetscReal thresholdForReducingOrder;

  PetscBool BondAssociativeModeling;

  PetscReal initialTime; //Initial time of our computation. usually 0
  PetscReal finalTime; //Final time of our computation. Problem dependent
  PetscReal currentTime; //Time currently (time in the time step we are in)
  PetscReal timeStep; //time step, problem and mesh dependent
  PetscInt  stepNumber; //number of the current step

  PetscReal youngModulus;
  PetscReal poissonRatio;
  PetscReal density;
  PetscReal lambda; //Lame parameter
  PetscReal mu; //Lame parameter
  PetscReal kappa;

  PetscReal SigmaYinitial; //The initial yield stress under which the material starts plasticizing. If we don't have hardening then this is constant
  PetscReal hardeningModulos; //Hardening modulus

  PetscReal supportFactor; //Ratio of support size to grid spacing
  PetscReal neighborSearchFactor; //Ratio of search bucket to support size. A bigger radius is used to form neighbor sets only once, in this semi-Lagrangian setting
  PetscInt  FreqResults; //Frequency with wich we export results for post processing
  PetscInt  numPoints; //total number of particles
  PetscInt  numFluidNodes;
  PetscReal Alpha_f; //parameter for generalized aplha method
  POINTS    *puntos; //structure POINT is a member of PARAMETERS

} PARAMETERS;

typedef struct {
  PetscScalar rho,ux,uy,temp;
} Field;

//// End Data structures for immersed-IGA //// 

//// I/O ////



//// End I/O ////




int main(int argc, char *argv[]) {

  // Initialize MPI and timer
  Teuchos::RCP<Epetra_Comm> epetraComm;
  int mpi_id = 0;
  int mpi_size = 1;
  #ifdef HAVE_MPI
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_id);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // conversion to epetra communicator takes place inside the Peridigm constructor now.
    epetraComm = Teuchos::RCP<Epetra_Comm>(new Epetra_MpiComm(MPI_COMM_WORLD));
  #else
    epetraComm = Teuchos::RCP<Epetra_Comm>(new Epetra_SerialComm);
  #endif

  // Banner
  if(mpi_id == 0){
    cout << "\n--Peridigm-ImmersedIGA--\n" << endl ;
    if(mpi_size > 1)
      cout << "MPI initialized on " << mpi_size << " processors.\n" << endl;
  }

  // Create a parameter list that will be passed to the Peridigm object
  Teuchos::RCP<Teuchos::ParameterList> peridigmParams(new Teuchos::ParameterList);
  RCP<ParameterList> discParams = rcp(new ParameterList);
  

  Teuchos::ParameterList& discretizationParams = peridigmParams->sublist("Discretization");
  discretizationParams.set("Type", "Text File");
  discretizationParams.set("Input Mesh File", "CylinderBlast1.txt");
  discParams->set("Type", "Text File");
  discParams->set("Input Mesh File", "CylinderBlast1.txt");

  Teuchos::ParameterList& materialParams = peridigmParams->sublist("Materials");
  materialParams.sublist("My Elastic Material");
  materialParams.sublist("My Elastic Material").set("Material Model", "Elastic");
  materialParams.sublist("My Elastic Material").set("Apply Shear Correction Factor", false);
  materialParams.sublist("My Elastic Material").set("Density", 7800.0);
  materialParams.sublist("My Elastic Material").set("Bulk Modulus", 130.0e9);
  materialParams.sublist("My Elastic Material").set("Shear Modulus", 78.0e9);

  // These have to be consistent, otherwise there will be a conflict with disc and then peridigm obj.
  ParameterList blockParameterList;
  ParameterList& blockParams = blockParameterList.sublist("My Block");
  blockParams.set("Block Names", "block_1");
  blockParams.set("Material", "My Elastic Material");
  blockParams.set("Horizon", 0.5);

  Teuchos::ParameterList& BlockParams = peridigmParams->sublist("Blocks");
  BlockParams.sublist("My Block");
  BlockParams.sublist("My Block").set("Block Names", "block_1");
  BlockParams.sublist("My Block").set("Material", "My Elastic Material");
  BlockParams.sublist("My Block").set("Horizon", 0.5);
  //

  Teuchos::ParameterList& outputParams = peridigmParams->sublist("Output");
  outputParams.set("Output File Type", "ExodusII");
  outputParams.set("NumProc", mpi_size);
  outputParams.set("Output Filename", "CylinderBlast1");
  outputParams.set("Final Output Step", 10);
  outputParams.set("Output Frequency", 1);
  outputParams.sublist("Output Variables");
  outputParams.sublist("Output Variables").set("Element_Id", true);
  outputParams.sublist("Output Variables").set("Block_Id", true);
  outputParams.sublist("Output Variables").set("Horizon", true);
  outputParams.sublist("Output Variables").set("Volume", true);
  outputParams.sublist("Output Variables").set("Model_Coordinates", true);
  outputParams.sublist("Output Variables").set("Coordinates", true);
  outputParams.sublist("Output Variables").set("Displacement", true);
  outputParams.sublist("Output Variables").set("Velocity", true);
  outputParams.sublist("Output Variables").set("Acceleration", true);
  outputParams.sublist("Output Variables").set("Force_Density", true);
  outputParams.sublist("Output Variables").set("Damage", true);
  outputParams.sublist("Output Variables").set("Proc_Num", true);


  PeridigmNS::HorizonManager::self().loadHorizonInformationFromBlockParameters(blockParameterList);

  cout<<"Making Discretization object\n"<<endl;
  // Create a discretization
  Teuchos::RCP<PeridigmNS::Discretization> textDiscretization(new PeridigmNS::TextFileDiscretization(epetraComm, discParams));
  cout<<"Created Discretization object\n"<<endl;

  // Create a Peridigm object
  cout<<"Making Peridigm object\n"<<endl;
  Teuchos::RCP<PeridigmNS::Peridigm> peridigm(new PeridigmNS::Peridigm(MPI_COMM_WORLD, peridigmParams, textDiscretization));
  cout<<"Created Peridigm object\n"<<endl;
  // Get RCPs to important data fields
  Teuchos::RCP<Epetra_Vector> initialPosition = peridigm->getX();
  Teuchos::RCP<Epetra_Vector> currentPosition = peridigm->getY();
  Teuchos::RCP<Epetra_Vector> displacement = peridigm->getU();
  Teuchos::RCP<Epetra_Vector> velocity = peridigm->getV();
  Teuchos::RCP<Epetra_Vector> force = peridigm->getForce();

  // Set the time step for Peridigm routines
  double myTimeStep = 0.1;
  peridigm->setTimeStep(myTimeStep);

  // apply strain
  for(int i=0 ; i<currentPosition->MyLength() ; i+=3){
    (*currentPosition)[i]   = 1.01 * (*initialPosition)[i];
    (*currentPosition)[i+1] = (*initialPosition)[i+1];
    (*currentPosition)[i+2] = (*initialPosition)[i+2];
  }

  // Set the displacement vector
  for(int i=0 ; i<currentPosition->MyLength() ; ++i)
    (*displacement)[i]   = (*currentPosition)[i] - (*initialPosition)[i];

  peridigm->writePeridigmSubModel(0);

  // Evaluate the internal force
  peridigm->computeInternalForce();

  
  // Assume we're happy with the internal force evaluation, update the state
  peridigm->updateState();

  //Write result to an ExodusII file: 
  peridigm->writePeridigmSubModel(1);

  // Write to stdout
  int colWidth = 10;
  if(mpi_id == 0){

    cout << "Initial positions:" << endl;
    for(int i=0 ; i<initialPosition->MyLength() ;i+=3)
      cout << "  " << setw(colWidth) << (*initialPosition)[i] << ", " << setw(colWidth) << (*initialPosition)[i+1] << ", " << setw(colWidth) << (*initialPosition)[i+2] << endl;

    cout << "\nDisplacements:" << endl;
    for(int i=0 ; i<displacement->MyLength() ; i+=3)
      cout << "  " << setw(colWidth) << (*displacement)[i] << ", " << setw(colWidth) << (*displacement)[i+1] << ", " << setw(colWidth) << (*displacement)[i+2] << endl;

    cout << "\nCurrent positions:" << endl;
    for(int i=0 ; i<currentPosition->MyLength() ; i+=3)
      cout << "  " << setw(colWidth) << (*currentPosition)[i] << ", " << setw(colWidth) << (*currentPosition)[i+1] << ", " << setw(colWidth) << (*currentPosition)[i+2] << endl;

    cout << "\nForces:" << endl;
    for(int i=0 ; i<force->MyLength() ; i+=3)
      cout << "  " << setprecision(3) << setw(colWidth) << (*force)[i] << ", " << setw(colWidth) << (*force)[i+1] << ", " << setw(colWidth) << (*force)[i+2] << endl;

    cout << endl;
  }

#ifdef HAVE_MPI
  MPI_Finalize() ;
#endif

  return 0;
}
