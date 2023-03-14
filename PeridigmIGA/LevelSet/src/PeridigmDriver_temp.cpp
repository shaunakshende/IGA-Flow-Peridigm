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
#include <iomanip>
#include <petscblaslapack.h>
#include <petsc/private/tsimpl.h>


// Boost parallelization libraries
#include <boost/graph/use_mpi.hpp>
#include <boost/graph/distributed/mpi_process_group.hpp>
#include <boost/graph/distributed/adjacency_list.hpp>
#include <boost/graph/distributed/distributed_graph_utility.hpp>
#include <boost/graph/distributed/local_subgraph.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/version.hpp>


// Standard Library includes
#include <fstream>
#include <string>
#include <math.h>
#include <vector>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

// Eigen Includes
#include <Eigen/Dense>

using namespace std;
using namespace Teuchos;
using namespace PeridigmNS;
using namespace Eigen;

#define SQ(A) ((A)*(A))
#define for1(i,n) for(int i=0; i<(n); i++)
#define for2(i,j,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++)
#define for3(i,j,k,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++)
#define for4(i,j,k,l,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++)
#define for5(i,j,k,l,m,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++)
#define for6(i,j,k,l,m,o,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++) for(int o=0; o<(n); o++)
#define for7(i,j,k,l,m,o,p,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++) for(int o=0; o<(n); o++) for(int p=0; p<(n); p++)
#define for8(i,j,k,l,m,o,p,q,n) for(int i=0; i<(n); i++) for(int j=0; j<(n); j++) for(int k=0; k<(n); k++) for(int l=0; l<(n); l++) for(int m=0; m<(n); m++) for(int o=0; o<(n); o++) for(int p=0; p<(n); p++) for(int q=0; q<(n); q++)

// Quantities that will be reduced into through MPI and Broadcast
int mpiSize;
double totalInitialExplosiveVolume;
double totalCurrentExplosiveVolume;
double totalExplosiveMass;
double totalExplosiveVolume;
int totalNumNodes;
int num_PD_nodes;

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
  IGA iga_LS;
  IGA iga_LSC;
  PetscReal Lx,Ly,Lz,mu,lamda,kappa,temp0,p0,R,Cp,Cv,spacing;
  Vec A1,Ap,An,V1,Vp,Vn,D1,Dp,Dn,Aa,Va,Da,V0,A0,D0,dA,density;
  Vec A1_LS,Ap_LS,An_LS,V1_LS,Vp_LS,Vn_LS,D1_LS,Dp_LS,Dn_LS,Aa_LS,Va_LS,Da_LS,V0_LS,A0_LS,D0_LS,dA_LS,PhiReference;
  Vec V0_LSC,A0_LSC,D0_LSC,dA_LSC;
  PetscInt  Nx,Ny,Nz,max_its;
  Mat Ainv;
  Mat MFinv;

  PetscInt *processor_numElX;
  PetscInt *processor_numElY;
  PetscInt *processor_numElZ;
  PetscReal *xhat;
  PetscReal *yhat;
  PetscReal *zhat;

  PetscReal totalInitialExplosiveVolume;
  PetscReal totalCurrentExplosiveVolume;
  PetscReal totalExplosiveMass;
  PetscReal TimeRestart;
  PetscInt  numFluidNodes;

  PetscInt  StepRestart;
  PetscInt  stepNumber;
  PetscInt  it;
  PetscInt  FreqRestarts;
  PetscInt  nen;
  PetscReal *GlobalForces;
  PetscReal *GlobalVel;
  PetscReal *GlobalDisp;
  PetscReal *COORD;
  PetscReal *VOLUME;
  PetscReal *NORMAL;
  //PD Nodes on which the boundary conditions are enforeced. These will be read from node_list files
  //ID's in these files correspond to fd PD_ID's, so anytime a bc needs to be set, the fd.PD_ID and the PD_ID_BC
  //can be checked.
  PetscInt  *PD_ID_BC;

  PetscReal thickness;
  PetscReal Alpha_m,Alpha_f,Gamma,Beta;
  PetscInt num_Owned_Points;
  PetscReal horizon;

  //Values for Peridigm restart files
  PetscReal PDInitialTime;
  PetscReal OutputRestart;
  PetscBool debug;

  // new quantities to incorporate LS fields
  PetscInt numLS_Fields;

} AppCtx;

// info to identify a vertex; need to be able to get unique ID on the fly
// w/out communication
class VertexID{
public:
  // three-part ID, using current processor rank, the rank of
  // the processor that created the vertex, and the local ID.  birthCert
  // and localID uniquely identify the vertex.  includeing current rank
  // in the VertexID and then returning it as a hash is the only way
  // I could find to explicitly control which process the vertex goes to.
  int rank;
  int birthCert;
  int localID;

  // in calls to constructors, birthCert_ should ALWAYS be the calling
  // task's MPI rank.  rank_ can be any other task, and determines
  // what task will own the new vertex created with this VertexID

  // constructor: updates local count automatically
  VertexID(int rank_, int birthCert_, int *localCount){
    rank = rank_;
    birthCert = birthCert_;
    localID = (*localCount);
    (*localCount)++;
  } // end constructor

  // constructor: does not update local count
  VertexID(int rank_, int birthCert_, int local_){
    rank = rank_;
    birthCert = birthCert_;
    localID = local_;
  } // end non-updating constructor

  // default constructor
  VertexID(){}

  void doCopy(const VertexID &id){
    rank = id.rank;
    birthCert = id.birthCert;
    localID = id.localID;
  } // end doCopy

  // destructor
  ~VertexID(){}

  // assignment
  VertexID &operator=(const VertexID &id){
    doCopy(id);
  } // end assignment

  template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int version){
    ar & rank & birthCert & localID;
  } // end serialize
}; // end VertexID

BOOST_IS_MPI_DATATYPE(VertexID);
// Particle information (this could be part of FieldData instead, but the idea
// is that this is "small" and contains less stuff that everything we
// need for computations at a particle, e.g., things used in decisions
// about graph connectivity, iteration, activity/inactivity, etc.)
class ParticleInfo{
public:
  // REMEMBER: if new field added, also add to serialize method
  double currentCoord[3];
  double initialCoord[3];
  double tempCoord[3];
  bool   isTask;

  ParticleInfo(){
    for1(i,3)
      currentCoord[i] = 0.0;
    isTask = false;
  } // end construtor

  ~ParticleInfo(){}

  template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int version){
    ar & currentCoord & initialCoord & tempCoord & isTask;
  } // end serialize
}; // end class ParticleInfo

BOOST_IS_MPI_DATATYPE(ParticleInfo);

// Holds state information for a particle; idea is that this is larger in
// memory than ParticleInfo, and holds all sorts of tensors, etc., that are
// only really needed in constitutive routines
class FieldData{
public:
  // REMEMBER: if new field added, also add to serialize method

  // Variables associated with assembly of foreground PD residual and
  // Peridigm integration
  double inertia[3];
  double residual[3];
  double internalForce[3];
  double bodyForce[3];
  double normal[3];
  double referenceNodalVolume;
  double alphaNodalVolume;
  double alphaNodalDensity;
  double referenceDensity;
  //
  int material;

  double nodalVolume;

  //For quadrature volume update, new variables:
  double nodalDensity;
  double nodalDensityInitial;
  double nodalPressure;
  double nodalVolumeInitial;

  int    Boundary;
  double totalPhysicalDisplacement[3];
  double totalPhysicalVelocity[3];
  double totalStrain[6];
  double totalStress[6];
  double totalStress0[6];
  double totalStrain0[6];
  double ductile_threshold0;
  double ductile_threshold;
  double brittle_threshold0;
  double brittle_threshold;
  double damage;
  double damage0;
  double currentDeformationGradient[9];
  double DeformationGradientOld[9];
  double alphaDeformationGradient[9];
  double velocityGradient[9];
  double determinantCurrentDeformationGradient;
  double determinantAlphaDeformationGradient;

  double totalPhysicalDisplacementOldIteration[3];
  double totalPhysicalDisplacementOldStep[3];
  double totalPhysicalVelocityOldIteration[3];
  double totalPhysicalVelocityOldStep[3];
  double totalPhysicalAcceleration[3];
  double totalPhysicalAccelerationOldIteration[3];
  double totalPhysicalAccelerationOldStep[3];
  double AccelerationIncrement[3];
  double currentCoord[3];
  double referenceCoord[3];
  double computed_currentCoord[3];
  double computed_tempCoord[3];


  double ductile_energy;
  double brittle_energy;
  int    Inside; //Flag to state if the particle is inside the background computational domain. 1 for inside, 0 for outside
  int    ID;
  int    ID_PD;

  double effectiveStrainRate;

  int flag;
  int flag0;

  /// ### Penalty Associated Variables ### ///
  double interpolatedVelocity[3];
  double penaltyParameter;
  double referencePenaltyParameterInternal;
  double referencePenaltyParameterInertia;
  bool   flyingPoint;
  double penaltyForce[3];
  //////////////////////////////////////////

  ///// Storing projected quantities for efficiency /////
  //double field[5];
  //double map[]


  FieldData(){}
  ~FieldData(){}

  template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int version){
    ar & interpolatedVelocity & penaltyParameter & referencePenaltyParameterInternal & referencePenaltyParameterInertia & flyingPoint & penaltyForce & nodalVolume & nodalDensity & nodalDensityInitial
  & nodalPressure & nodalVolumeInitial & Boundary & totalPhysicalDisplacement & totalPhysicalVelocity & totalStrain & totalStress & totalStrain0 & totalStress0
  & ductile_threshold0 & ductile_threshold & brittle_threshold0 & brittle_threshold & damage0 & damage & currentDeformationGradient & velocityGradient
  & determinantCurrentDeformationGradient & totalPhysicalDisplacementOldIteration & totalPhysicalDisplacementOldStep & totalPhysicalVelocityOldIteration
  & totalPhysicalVelocityOldStep & totalPhysicalAcceleration & totalPhysicalAccelerationOldIteration & totalPhysicalAccelerationOldStep & AccelerationIncrement
  & ductile_energy & brittle_energy & Inside & DeformationGradientOld & alphaDeformationGradient & determinantAlphaDeformationGradient & effectiveStrainRate
  & flag & flag0 & inertia & residual & bodyForce & normal & internalForce & referenceNodalVolume & alphaNodalDensity & alphaNodalVolume & ID & ID_PD & material & referenceDensity
  & referenceCoord & currentCoord & computed_currentCoord & computed_tempCoord;
  } // end serialize
}; // end class fieldData

BOOST_IS_MPI_DATATYPE(FieldData);

// class to hold all of the stuff at a graph vertex
class VertexData{
public:

  // the approach i've taken is to store parts of data in other classes, then
  // have instances of those classes as members.  the reason i do this instead
  // of storing everything as members of VertexData is that, to access
  // members of VertexData from vertex descriptors (typedef'd as "Vertex")
  // we'd need a new property map for each one.  this would make
  // adding/removing new fields/data associated with particles very tedious.

  // This exists to uniquely-identify a vertex
  VertexID id;

  // this stores a small amount of info used by things other than
  // constitutive routines
  ParticleInfo info;

  // this stores all of the physical data associated with a point (except
  // for its location, which is used in determining graph connectivity,
  // and is stored in info)
  FieldData fd;

  // given ID only
  VertexData(VertexID id_){
    id = id_;
    info = ParticleInfo();
    fd = FieldData();
  } // end ID only constructor

  // given ID, field data
  VertexData(VertexID id_, ParticleInfo info_){
    id = id_;
    info = info_;
    fd = FieldData();
  } // end ID only constructor

  VertexData(){}
  ~VertexData(){}

  template<typename Archiver>
  void serialize(Archiver& ar, const unsigned int version){
    ar & id & fd & info;
  } // end serialize

}; // end VertexData

BOOST_IS_MPI_DATATYPE(VertexData);

// hash function for VertexID class
// appaerently this determines which process a named vertex goes to
// (you'd think the docs would say this somewhere...)
std::size_t hash_value(const VertexID &v) {
  //return boost::hash_value((size_t)v.localID + (size_t)v.rank);
  return (size_t)v.rank + ((size_t)mpiSize)*((size_t)v.localID);
  //return (size_t)v.rank;
} // end hash function

bool operator==(const VertexID &id1, const VertexID &id2){
  return (id1.rank==id2.rank)&&(id1.localID==id2.localID)
    &&(id1.birthCert==id2.birthCert);
} // end overloaded ==

// This snippet is used to allow acces of vertices through their VertexID
// attributes
namespace boost { namespace graph {
    template<>
    struct internal_vertex_name<VertexData>
    {
      typedef multi_index::member<VertexData, VertexID, &VertexData::id> type;
    };
  }
} // end activating named vertices

// This specifies what to do if a vertex is requested via ID, but no such
// vertex exists
namespace boost { namespace graph {
    template<>
    struct internal_vertex_constructor<VertexData>
    {
      typedef vertex_from_name<VertexData> type;
    };
  }
} // end allowing new verticies

using namespace boost;
using boost::graph::distributed::mpi_process_group;
// there is an apparent bug in the parallel boost graph library that causes
// calls to the existing remove_vertex() function to cause
// compiler errors, so I introduced my own version that comments out the
// offending line.  No idea whether this could potentially make trouble, but
// it seems to work okay...
template<PBGL_DISTRIB_ADJLIST_TEMPLATE_PARMS>
void
remove_vertex_dk(typename PBGL_DISTRIB_ADJLIST_TYPE::vertex_descriptor u,
        PBGL_DISTRIB_ADJLIST_TYPE& graph)
{
  typedef typename PBGL_DISTRIB_ADJLIST_TYPE::graph_type graph_type;
  typedef typename graph_type::named_graph_mixin named_graph_mixin;
  BOOST_ASSERT(u.owner == graph.processor());
  static_cast<named_graph_mixin&>(static_cast<graph_type&>(graph))
    .removing_vertex(u, boost::graph_detail::iterator_stability
         (graph.base().m_vertices));

  // this line caused an error ///////
  //g.distribution().clear();
  ////////////////////////////////////

  remove_vertex(u.local, graph.base());
}

// bidirectional graph, using a list for the underlying data structure
typedef adjacency_list<listS, // out edge list type
           distributedS<mpi_process_group, listS>, // vertex list
           //bidirectionalS,
           directedS, // graph type
           VertexData, // vertex property
           no_property, // edge property
           no_property, // graph property
           listS> // edge list
Graph;

// renaming convoluted boost types for easier use
typedef graph_traits<Graph>::vertex_descriptor Vertex;
typedef graph_traits<Graph>::edge_descriptor Edge;
typedef graph_traits<Graph>::vertex_iterator VertexIterator;
typedef graph_traits<Graph>::adjacency_iterator AdjIterator;
typedef graph_traits<Graph>::edge_iterator EdgeIterator;
typedef graph_traits<Graph>::out_edge_iterator OutEdgeIterator;

// these allow us to access members of the vertex property type VertexData
// given a vertex descriptor
property_map<Graph, VertexID VertexData::*>::type
id_property;
property_map<Graph, ParticleInfo VertexData::*>::type
info_property;
property_map<Graph, FieldData VertexData::*>::type
fd_property;

// class that wraps around Graph and provides access to property maps
// and methods for refinement and coarsening
class ParticleManager{
public:
  Graph graph;
  int localVertexCounter;
  int myRank;
  int nProc;
  Vertex myTaskVertex;
  vector<Vertex> taskVertexDescriptors;

  // copy constructor
  ParticleManager(const ParticleManager &manager){}

  void generatePropertyMaps(){
    id_property = get(&VertexData::id, graph);
    id_property.set_consistency_model
      (boost::parallel::consistency_model::cm_bidirectional);
    info_property = get(&VertexData::info, graph);
    info_property.set_consistency_model
      (boost::parallel::consistency_model::cm_bidirectional);
    fd_property = get(&VertexData::fd, graph);
    fd_property.set_consistency_model
      (boost::parallel::consistency_model::cm_bidirectional);
  } // end generatePropertyMaps

  // synchronize processors
  void sync(){
    synchronize(graph.process_group());
  } // end sync

  // constructor
  ParticleManager(){

    generatePropertyMaps();

    myRank = graph.process_group().rank;
    nProc = graph.process_group().size;
    taskVertexDescriptors = vector<Vertex>();

    // add graph vertices corresponding to processor subdomains
    for1(rank,nProc){
      // dummy local vertex counter
      int dummyCounter = 0;
      // create VertexID that hashes to rank and has local vert number or zero
      VertexID id = VertexID(rank,rank,&(dummyCounter));
      // create vertex from VertexID
      VertexData vd = VertexData(id);
      vd.info.isTask = true;
      // actually add vertex to graph if mine
      if(rank == myRank){
  // keep track of my task vertex descriptor
      myTaskVertex = add_vertex(vd,graph);}
    } // rank
    sync();

    // add descriptors to task vertexes
    for1(rank,nProc){
      // dummy local vertex counter
      int dummyCounter = 0;
      // create VertexID that hashes to rank and has local vert number or zero
      VertexID id = VertexID(rank,rank,&(dummyCounter));
      // add corresponding descriptor to list of task verts on all ranks
      taskVertexDescriptors.push_back(*find_vertex(id,graph));
    } // rank
    sync();

    // set local counter to 1 to reflect task vert added
    localVertexCounter = 1;

  } // end constructor

  // obtain, in O(N) time, the rank of the processor whose subdomain contains
  // the point
  int pointRank(double *x, AppCtx *user){
    if(user->debug){
      x[0] = x[0]*user->Lx;
      x[1] = x[1]*user->Ly;
      x[2] = x[2]*user->Lz;
    }
	  PetscInt elemX,elemY,elemZ, e, m;
    elemX = 0; elemY = 0; elemZ = 0, e=-1;
	  int i,j;
    PetscScalar *U;

    m = user->iga->axis[0]->m;
    U = user->iga->axis[0]->U;
    for(j=0;j<m;j++){
     if(U[j+1]-U[j]>1.0e-13) e += 1;
     if(x[0]/user->Lx >= U[j] && x[0]/user->Lx < U[j+1] && x[0]>0.0) elemX = e;
    }

    e = -1;
    m = user->iga->axis[1]->m;
    U = user->iga->axis[1]->U;
    for(j=0;j<m;j++){
     if(U[j+1]-U[j]>1.0e-13) e += 1;
     if(x[1]/user->Ly >= U[j] && x[1]/user->Ly < U[j+1] && x[1] > 0.0) elemY = e;
    }

    e = -1;
    m = user->iga->axis[2]->m;
    U = user->iga->axis[2]->U;
    for(j=0;j<m;j++){
     if(U[j+1]-U[j]>1.0e-13) e += 1;
     if(x[2]/user->Lz >= U[j] && x[2]/user->Lz < U[j+1] && x[2] > 0.0) elemZ = e;
    }

    // if((x[2]-0.07500)*(x[2]-0.07500)<1.0e-13){
    //   PetscPrintf(PETSC_COMM_SELF, "ElemX calculated = %d, ElemY calculated = %d , ElemZ calculated = %d  \n", elemX, elemY, elemZ);
    // }
    //Assemble "rank" in each direction, or the processor associated with the knot span that the particle
    // occupies
	  PetscInt rankX=-1,rankY=-1,rankZ=-1;
	  PetscInt check = -1;
	  for (i=0;i<user->iga->proc_sizes[0];i++){
		  check += user->processor_numElX[i];
		  if (elemX<=check){
			  rankX = i;
			  break;
		  }
	  }
	  check = -1;
	  for (i=0;i<user->iga->proc_sizes[1];i++){
		  check += user->processor_numElY[i];
		  if (elemY<=check){
			  rankY = i;
			  break;
		  }
	  }
	  check = -1;
	  for (i=0;i<user->iga->proc_sizes[2];i++){
		  check += user->processor_numElZ[i];
		  if (elemZ<=check){
			  rankZ = i;
			  break;
		  }
	  }

    //Find global rank
    PetscInt myRank = rankZ*user->iga->proc_sizes[0]*user->iga->proc_sizes[1] + rankY*user->iga->proc_sizes[0] + rankX;

    if ((elemX < 0) || (elemY < 0) || (elemZ < 0) || rankX < 0 || rankY < 0 || rankZ < 0) myRank = -1;
    if ((elemX >= user->iga->elem_sizes[0]) || (elemY >= user->iga->elem_sizes[1]) || (elemZ >= user->iga->elem_sizes[2])) myRank = -1;
    if(myRank<0){PetscPrintf(PETSC_COMM_SELF, "Element containing point not found! Exiting...\n");
    exit(1);}
    return myRank;
  }


  // connect particles to their corresponding task vertexes
  // "reconnect" is whether or not verts have previously been connected
  void connectVertsToTasks(bool reconnect,AppCtx *user){

    // vector to store information about what edges to add
    vector<pair<Vertex,Vertex>> edgesToAdd = vector<pair<Vertex,Vertex>>();

    // if we're just updating existing connectivity:
    if(reconnect){
      // iterate over all out-edges of this task's vertex.  delete ones
      // that are no longer valid, and make a note (in edgesToAdd) to add
      // any new edges that are needed.
 pair<OutEdgeIterator,OutEdgeIterator> its
  = out_edges(myTaskVertex,graph);
      auto it=its.first;
      while(it != its.second){
  Edge edge = *it;
  // vertex descriptor for particle on the other end of the edge
  Vertex v = target(edge,graph);
  // get the vertex's position, and use it to decide whether or not it
  // is still located in this processor's subdomain
  ParticleInfo info = get(info_property,v);
  int pr = pointRank(&info.currentCoord[0],user);
  // if the particle is no longer located on this processor's subdomain
  if(pr != myRank){
    // make a note to add a new edge from the task whose subdomain
    // the particle is located in.
          if (pr >= 0){
    edgesToAdd.push_back
      (pair<Vertex,Vertex>(taskVertexDescriptors[pr],v));
    }
    // remove the old edge:
    // note: removing an edge will invalidate the iterators pointing to
    // it, so we need to make a copy, advance the original, then
    // pass the copy to remove_edge()
    auto itTemp = it;
    ++it;
    // the version taking an iterator (instead of *itTemp) is faster
    // for directed graphs
    remove_edge(itTemp,graph);
  }else{
    // if the particle is still in this processor's subdomain, just
    // move on to the next edge
    ++it;
  } // end if
      } // it
    }else{ // otherwise, assume we're generating connectivity from scratch

      // remove existing edges (assume O(E/V) time...?)
      // if we're really only calling this on the first step, we don't need
      // to call clear_...(), but this branch is useful for debugging.
      clear_out_edges(myTaskVertex,graph);

      // loop over all vertices in the graph, determine which processor's
      // subdomain they are in, then make a note to add the corresponding
      // edge.
      BGL_FORALL_VERTICES(v,graph,Graph){
         ParticleInfo info = get(info_property,v);
           if(!info.isTask){
    // add edge from task to vertex
    edgesToAdd.push_back
      (pair<Vertex,Vertex>
       (taskVertexDescriptors[pointRank(&info.currentCoord[0],user)],v));
  } // end if is a particle, not a task
      } // v
    } // end switch on reconnecting
    sync();

    // actually create new edges from information stored in edgesToAdd

    for1(i,edgesToAdd.size()){
      add_edge(edgesToAdd[i].first,edgesToAdd[i].second,graph);
    } // i
    sync();

    // generatePropertyMaps();
    // sync();
    //PetscPrintf(PETSC_COMM_WORLD,"Return connectVertsToTasks");
  } // end connectVertsToTasks
}; // end ParticleManager class

#define SQ(A) ((A)*(A))
typedef struct
{
    PetscReal initialTime;
    PetscReal finalTime;
    PetscReal currentTime;
    PetscReal gamma;
    PetscReal beta;
    PetscReal timeStep;
    PetscInt  stepNumber;

    PetscReal max_compressive_strain;
    PetscReal max_tensile_strain;
    PetscReal max_compressive_stress;
    PetscReal max_tensile_stress;
    PetscReal brittle_initial_threshold;
    PetscReal ductile_initial_threshold;

    PetscReal   youngModulus;
    PetscReal   poissonRatio;
    PetscReal   density;
    PetscReal   lambda;
    PetscReal   mu;
    PetscReal   massDamping;

    PetscReal   SigmaYinitial;  //Plasticity

    PetscInt    FreqResults;
    PetscInt    numPoints;
    PetscInt    numNodes;
    PetscReal   Alpha_f;

    PetscReal densityRDX;

    PetscInt rateEffects;

    PetscReal penaltyConstant;
    PetscBool DamageModeling;
    PetscReal damageCriticalStress;
    PetscReal damageCriticalEpsilonPlastic;
    PetscReal thresholdDamageForPenalty;

    PetscReal timeTildeFactor;

} PARAMETERS;

typedef struct {
  PetscScalar rho, ux, uy, uz, temp, alpha;
} Field;

typedef struct {
  PetscScalar phi;
} LS_Field; // {Level set variable alone for RD}

typedef struct {
  PetscScalar phi_prime;
} LSC_Field; // {Level set variable correction constant}

//// End Data structures for immersed-IGA ////

////////// Missing Utilities //////////////////
PetscBool IGAElementNextFormIFunction(IGAElement element,IGAFormIFunction *fun,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *fun = form->ops->IFunction;
  *ctx = form->ops->IFunCtx;
  return PETSC_TRUE;
}

#undef  __FUNCT__
#define __FUNCT__ "IGAGetElementDiagonal"
PetscReal IGAGetElementDiagonal(IGA iga, PetscReal *pnt, AppCtx *user)
{

  PetscInt    i,j,m,dim=iga->dim;
  PetscScalar *U;
  PetscReal   Lengths[3] = {0.0};
  Lengths[0] = user->Lx;
  Lengths[1] = user->Ly;
  Lengths[2] = user->Lz;

  PetscReal diagonal = 0.0;
  int check = 0;
  int temporary = 0;

  for(i=0;i<dim;i++){
    U = iga->axis[i]->U;
    m = iga->axis[i]->m;

   /* find which nonzero span this point is located in and calculate square of side length and add to diagonal^2 */
    for(j=0;j<m;j++){
     if(pnt[i] >= U[j] && pnt[i] < U[j+1] || abs(pnt[i]-1.0)<1.0e-13){
       if(abs(pnt[i]-1.0)>=1.0e-13){
       diagonal += (U[j+1]-U[j])*(U[j+1]-U[j])*Lengths[i]*Lengths[i];
       check++;
       temporary++;
       }
       if(abs(pnt[i]-1.0)<1.0e-13){
         diagonal += (U[m-2]-U[m-3])*(U[m-2]-U[m-3])*Lengths[i]*Lengths[i];
         check++;
         break;
       }
       break;
    }
  }
}

  diagonal = sqrt(diagonal);
  if(check !=dim && dim>2){
    PetscPrintf(PETSC_COMM_SELF, "check = %d, Could not find knot span surrounding point! diag = %e ; pt = %e %e %e\n", check, diagonal, pnt[0], pnt[1], pnt[2]);
    PetscPrintf(PETSC_COMM_SELF, "Condition 1 = %d\n", temporary);
    PetscPrintf(PETSC_COMM_SELF, "Lengths = %e %e %e\n", Lengths[0], Lengths[1], Lengths[2]);
    PetscPrintf(PETSC_COMM_SELF, "Conditions evaluated = %d %d %d || %d %d %d\n", abs(pnt[0]-1.0)<1.0e-13, abs(pnt[1]-1.0)<1.0e-13, abs(pnt[2]-1.0)<1.0e-13,
    pnt[0]<1.0, pnt[1]<1.0, pnt[2]<1.0);
    exit(1);
  }
  if(check !=dim && dim==2){
    PetscPrintf(PETSC_COMM_SELF, "Could not find knot span surrounding point! diag = %e pt = %e %e\n", diagonal, pnt[0], pnt[1]);
    PetscPrintf(PETSC_COMM_SELF, "Lengths = %e %e\n", Lengths[0], Lengths[1]);
    exit(1);
  }

  return diagonal;
}
///////////////////////////////////////////////

/////// Level Set functions //////////////////////
#undef  __FUNCT__
#define __FUNCT__ "alpha"
PetscReal alpha(AppCtx *user, PetscReal phi, PetscReal localWidth){
PetscFunctionBegin;
PetscReal out = 0.5*(tanh(phi/(localWidth))/0.987+1.0);
PetscFunctionReturn(out);
}

#undef  __FUNCT__
#define __FUNCT__ "regularizedSignFunction"
PetscReal regularizedSignFunction(AppCtx *user, PetscReal phi, PetscReal epsilon){
// Regularized Sign function. Provide user, phi, and local interface width
PetscFunctionBegin;
PetscReal out = tanh(phi/epsilon)/0.987;
PetscFunctionReturn(out);
}

#undef  __FUNCT__
#define __FUNCT__ "TaitEq_WATER"
PetscReal TaitEq_WATER(PetscReal density, PetscReal* fprime, PetscReal* cs){
//Evaluate tait equation to get pressure
PetscFunctionBegin;

// Tait Eq. Water EOS properties (should be in a function)
PetscReal dens0  = 1000.0;
PetscReal P0     = 100000.0;
PetscReal B      = 3.31e8;
PetscReal N      = 7.15;
PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
PetscReal Pcr    = 22.02726;
//
PetscReal P        = Pcr;
*fprime   = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
*cs       = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
if(density>rhoCR){
   P         = P0+B*(pow(density/dens0,N))-B;
   *fprime   = (1.0/density)*B*N*(pow(density/dens0,N));
   *cs       = (1.0/density)*B*N*(pow(density/dens0,N));
 }
PetscFunctionReturn(P);
}

#undef  __FUNCT__
#define __FUNCT__ "Energy_WATER"
PetscReal Energy_WATER(PetscReal T){
//Evaluate reduced energy equation to get energy in water
PetscFunctionBegin;
PetscReal E = 4184.0*T;
PetscFunctionReturn(E);
}

#undef  __FUNCT__
#define __FUNCT__ "JWL_RDX"
PetscReal JWL_RDX(PetscReal dens, PetscReal* fprime, PetscReal* cs){
//Evaluate tait equation to get pressure
PetscFunctionBegin;

PetscReal dens0  = 1770.0;
PetscReal P0     = 100000.0;
PetscReal A      = 7.78e11;
PetscReal B      = 7.07e9;
PetscReal C      = 1.62e9;
PetscReal R1     = 4.485;
PetscReal R2     = 1.068;
PetscReal omega  = 0.3;
PetscReal E0     = 5.93e6;
PetscReal Pcr    = 2.0e11;

PetscReal nu     = dens0/dens;
PetscReal Ptest  = A*exp(-R1*nu) + B*exp(-R2*nu) + C/(pow(nu, 1+omega)); //We can consider just one branch of the JWL
//Equation, as they intersect at 2E11, to determine if P>2E11. Then, we will choose the right function based
//on this value

PetscReal P        = Pcr;
*fprime   = 0.0;
*cs       = 0.0;
if(PETSC_FALSE){//Ptest > Pcr){ // We are not currently using this branch of the JWL
  exit(0);
  P        = Ptest;
  *fprime   = A*exp(-R1*nu)*R1*nu/dens +  B*exp(-R2*nu)*R2*nu/dens  +  C*(1+omega)*pow(dens,omega)/pow(dens0,1+omega);
  *cs       = *fprime;
}else{
  P        = A*(1-omega/(R1*nu))*exp(-R1*nu) + B*(1-omega/(R2*nu))*exp(-R2*nu) + omega*dens0*E0/nu;
  *fprime   = ((A*omega/(R1*dens0))*(-R1*nu-1)+nu*A*R1/dens)*exp(-R1*nu) + ((B*omega/(R2*dens0))*(R2*nu-1)+nu*B*R2/dens)*exp(-R2*nu) + omega*E0;

  *cs       = *fprime;
}

PetscFunctionReturn(P);
}

#undef  __FUNCT__
#define __FUNCT__ "Energy_RDX"
PetscReal Energy_RDX(PetscReal T){
//Evaluate reduced energy equation to get energy in RDX
PetscFunctionBegin;
PetscReal E = 143.3*T;
PetscFunctionReturn(E);
}


#undef __FUNCT__
#define __FUNCT__ "InitializeLS"
PetscErrorCode InitializeLS(IGA iga, IGA iga_LS,
                                    PetscReal t,
                                    Vec U,
                                    Vec U_LS,
                                    AppCtx *user,
                                    PARAMETERS *par)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  DM da;
  DM da_LS;

  PetscInt dof = iga->dof;
  PetscInt dof_LS = iga_LS->dof;
  PetscInt dim = iga->dim;
  PetscInt dim_LS = iga_LS->dim;

  ierr = IGACreateNodeDM(iga,dof,&da);CHKERRQ(ierr);
  Field ***u;
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = IGACreateNodeDM(iga_LS,dof_LS,&da_LS);CHKERRQ(ierr);
  LS_Field ***u_LS;
  ierr = DMDAVecGetArray(da_LS,U_LS,&u_LS);CHKERRQ(ierr);
  DMDALocalInfo info_LS;
  ierr = DMDAGetLocalInfo(da_LS,&info_LS);CHKERRQ(ierr);

  PetscInt i,j,k;
  PetscInt nodesX  = iga->geom_lwidth[0], nodesY  = iga->geom_lwidth[1], nodesZ  = iga->geom_lwidth[2];
  PetscInt gnodesX = iga->geom_gwidth[0], gnodesY = iga->geom_gwidth[1];
  PetscReal pt[3];

  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      for(k=info.zs;k<info.zs+info.zm;k++){
        PetscReal x = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim];
        PetscReal y = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 1];
        PetscReal z = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 2];

        pt[0] = x/user->Lx;
        pt[1] = y/user->Ly;
        pt[2] = z/user->Lz;

        PetscReal localWidth = IGAGetElementDiagonal(iga, pt, user);

        PetscScalar alpha_1 = u[k][j][i].alpha;
        if(u[k][j][i].alpha > 1.0){alpha_1 = 1.0;}
        if(u[k][j][i].alpha < 0.0){alpha_1 = 0.0;}
        u_LS[k][j][i].phi = (localWidth)*atanh(0.987*(2.0*alpha_1-1.0));

        }
      }
    }

  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da_LS,U_LS,&u_LS);CHKERRQ(ierr);
  ierr = DMDestroy(&da_LS);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "InitializeLSC"
PetscErrorCode InitializeLSC(IGA iga,
                             Vec U)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  PetscInt dim = iga->dim;

  ierr = IGACreateNodeDM(iga,1,&da);CHKERRQ(ierr);
  LSC_Field ***u;
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  for(int i=info.xs;i<info.xs+info.xm;i++){
    for(int j=info.ys;j<info.ys+info.ym;j++){
      for(int k=info.zs;k<info.zs+info.zm;k++){
        u[k][j][i].phi_prime  =  0.0;
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UpdatePhi"
PetscErrorCode UpdatePhi(IGA iga,
                         IGA iga_LS,
                         IGA iga_LSC,
                         Vec U,
                         Vec U_LS,
                         Vec U_LSC,
                         AppCtx *user,
                         PARAMETERS *par)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  DM da;
  DM da_LS;
  DM da_LSC;

  PetscInt dof     = iga->dof;
  PetscInt dof_LS  = iga_LS->dof;
  PetscInt dof_LSC = iga_LSC->dof;
  PetscInt dim     = iga->dim;
  PetscInt dim_LS  = iga_LS->dim;
  PetscInt dim_LSC = iga_LSC->dim;

  ierr = IGACreateNodeDM(iga,dof,&da);CHKERRQ(ierr);
  Field ***u;
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);

  ierr = IGACreateNodeDM(iga_LS,dof_LS,&da_LS);CHKERRQ(ierr);
  LS_Field ***u_LS;
  ierr = DMDAVecGetArray(da_LS,U_LS,&u_LS);CHKERRQ(ierr);
  DMDALocalInfo info_LS;
  ierr = DMDAGetLocalInfo(da_LS,&info_LS);CHKERRQ(ierr);

  ierr = IGACreateNodeDM(iga_LSC,dof_LSC,&da_LSC);CHKERRQ(ierr);
  LSC_Field ***u_LSC;
  ierr = DMDAVecGetArray(da_LSC,U_LSC,&u_LSC);CHKERRQ(ierr);
  DMDALocalInfo info_LSC;
  ierr = DMDAGetLocalInfo(da_LSC,&info_LSC);CHKERRQ(ierr);

  PetscReal pt[3];
  PetscInt i,j,k;
  PetscInt nodesX  = iga->geom_lwidth[0], nodesY  = iga->geom_lwidth[1], nodesZ  = iga->geom_lwidth[2];
  PetscInt gnodesX = iga->geom_gwidth[0], gnodesY = iga->geom_gwidth[1];

  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      for(k=info.zs;k<info.zs+info.zm;k++){
        PetscReal x = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim];
        PetscReal y = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 1];
        PetscReal z = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 2];

        pt[0] = x/user->Lx;
        pt[1] = y/user->Ly;
        pt[2] = z/user->Lz;

        PetscReal localWidth = IGAGetElementDiagonal(iga, pt, user);

        PetscScalar phi = u_LS[k][j][i].phi + u_LSC[k][j][i].phi_prime;
        u[k][j][i].alpha = 0.5*(tanh(phi/localWidth)/0.987+1.0);
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da_LS,U_LS,&u_LS);CHKERRQ(ierr);
  ierr = DMDestroy(&da_LS);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da_LSC,U_LSC,&u_LSC);CHKERRQ(ierr);
  ierr = DMDestroy(&da_LSC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
///////////////////////////////////////////////////////


//// Re-definition of PetIGA functions ////
#undef  __FUNCT__
#define __FUNCT__ "IGALocateElement_1"
PetscBool IGALocateElement_1(IGA iga,PetscReal *pnt,IGAElement element, AppCtx *user, ParticleManager &manager)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  PetscInt i,j,e,m,dim=iga->dim;
  PetscInt *ID = element->ID;
  PetscInt *width = element->width;
  PetscInt *start = element->start;
  PetscScalar *U;

  element->nen  = 1;
  element->nval = 0;
  element->nvec = 0;
  element->nmat = 0;
  for(i=0;i<dim;i++){
    element->nen *= (iga->axis[i]->p+1);
    U = iga->axis[i]->U;
    m = iga->axis[i]->m;
    e = -1;
    ID[i] = 0;
   /* find which nonzero span this point is located in */
    for(j=0;j<m;j++){
     if(U[j+1]-U[j]>1.0e-13) e += 1;
     if(pnt[i] >= U[j] && pnt[i] < U[j+1] && pnt[i] > 0.0) ID[i] = e;
    }
    /* reject if the element is not in this partition */
     if(ID[i] < iga->elem_start[i] || ID[i] >= iga->elem_start[i]+iga->elem_width[i]){
     PetscPrintf(PETSC_COMM_SELF, "Error: Particle on domain but not located by IGALocateElement! This Probably means that the rank of the point \n and the rank of the Subdomain assigned to rank are not consistent when this function is being called! Check pointRank. Exiting...\n");
     user->debug=PETSC_TRUE;
     int CalculatedRank = manager.pointRank(pnt, user);
     PetscPrintf(PETSC_COMM_SELF, "Rank is %d elemX = %d elemY = %d and elemZ = %d. Calculated Rank is %d!\n", rank, ID[0], ID[1], ID[2], CalculatedRank);
     MPI_Barrier(PETSC_COMM_WORLD);
     exit(1);}
}
  element->index = 0;
{
    PetscErrorCode ierr;
    ierr = IGAElementBuildClosure(element);CHKERRCONTINUE(ierr);
    if (PetscUnlikely(ierr)) PetscFunctionReturn(PETSC_FALSE);
    ierr = IGAElementBuildFix(element);CHKERRCONTINUE(ierr);
    if (PetscUnlikely(ierr)) PetscFunctionReturn(PETSC_FALSE);
  }

  return PETSC_TRUE;
}

PETSC_STATIC_INLINE
PetscBool IGAElementNextFormIJacobian(IGAElement element,IGAFormIJacobian *jac,void **ctx)
{
  IGAForm form = element->parent->form;
  if (!IGAElementNextForm(element,form->visit)) return PETSC_FALSE;
  *jac = form->ops->IJacobian;
  *ctx = form->ops->IJacCtx;
  return PETSC_TRUE;
}
//// End Re-Definition of PetIGA functions ////


//// I/O ////
#undef  __FUNCT__
#define __FUNCT__ "input"
// Reads foreground input files with format Pre-Processor V3:
//This function reads the input files for the particle data, specifically foreground(pr+1).dat
// Read file structured as:
// #nodes 0 0 0 0 0 0 0 0
// ID x y z dx dy dz material_ID vol
PetscErrorCode input (PARAMETERS *par,ParticleManager &manager, AppCtx *user)
{

  PetscFunctionBegin;
  //PetscPrintf(PETSC_COMM_WORLD," Input Begin");
  PetscErrorCode  ierr;
  PetscInt i,j;
  PetscInt num_proc;
  PetscInt temp;
  PetscReal tempR;
  int pr;

  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  //PetscPrintf(PETSC_COMM_WORLD," Processor number: %d", rank);


  ostringstream convert;
  convert << "foreground" << rank+1 << ".dat";
  string fname = convert.str();
  ifstream fin;
  fin.open(fname.c_str());
  if(fin.fail()){
    PetscPrintf(PETSC_COMM_WORLD,"foreground.dat file NOT found! Solving Fluid Only...\n");
    par->numPoints =  0;
    par->numNodes  =  0;
    PetscFunctionReturn(0);
  }

  fin >> par->numNodes;
  for (j = 0; j<6; j++){
    fin >> tempR; //Skip over the padding zeros in first line (8 zeros)
  }
  fin >> temp; // Material_ID field is an integer
  fin >> tempR;


  par->numPoints = par->numNodes;
 //PetscPrintf(PETSC_COMM_SELF," %d \n",par->numNodes);
  // add a bunch of verticies
            for1(i,par->numNodes){
              ParticleInfo info = ParticleInfo();
              FieldData fd = FieldData();
              fin >> fd.ID;//ID
              fin >> info.currentCoord[0];//x
              fin >> info.currentCoord[1];//y
              fin >> info.currentCoord[2];//z
              fin >> tempR;//x-width
              fin >> tempR;//y-width
              fin >> tempR;//z-width

              //Material Flag:
              fin >> fd.material;
              // Current definition:
              // 0 - Composite Shell
              // 1 - RDX
              // 2 - Air

              //initialize NodalVolume = x*y*z
              fin >> fd.nodalVolume;

              //Initialize reference and current quantities:
              info.initialCoord[0] = info.currentCoord[0];
              info.initialCoord[1] = info.currentCoord[1];
              info.initialCoord[2] = info.currentCoord[2];
              fd.computed_currentCoord[0] = info.currentCoord[0];
              fd.computed_currentCoord[1] = info.currentCoord[1];
              fd.computed_currentCoord[2] = info.currentCoord[2];

              fd.nodalVolumeInitial = 0.0;
              fd.nodalVolumeInitial += fd.nodalVolume;
              fd.referenceNodalVolume=fd.nodalVolumeInitial;

              //Debugging
              //PetscPrintf(PETSC_COMM_WORLD,"volI %e vol %e \n",fd.nodalVolume, fd.nodalVolumeInitial);


              fd.Boundary = 0;
              fd.Inside = 1;


              if(fd.material==0){
                fd.damage = 0;
                fd.flyingPoint = false;
                PetscReal meshSize = user->horizon/3.0;//0.305/151.0;//(par->puntos[i].support[0] + par->puntos[i].support[1])/(2.0*par->supportFactor); // average particle spacing
                fd.referencePenaltyParameterInternal = par->penaltyConstant * par->youngModulus * par->timeStep / (meshSize * meshSize);
                fd.referencePenaltyParameterInertia = par->penaltyConstant * par->density / par->timeStep;
                fd.penaltyParameter = fd.referencePenaltyParameterInternal;
              }

              for (j=0;j<6; j++){
                fd.totalStrain[j]  = 0.0;
                fd.totalStrain0[j] = 0.0;
                fd.totalStress[j]  = 0.0;
                fd.totalStress0[j] = 0.0;
              }

              for (j=0;j<3; j++){
                fd.totalPhysicalDisplacement[j] = 0.0;
                fd.totalPhysicalVelocity[j]     = 0.0;
                info.currentCoord[0] = info.currentCoord[0]+0.00000;
                info.currentCoord[1] = info.currentCoord[1]+0.00000;
                info.currentCoord[2] = info.currentCoord[2]+0.00000;
              }

              if (par->stepNumber == 0){
                    pr = manager.pointRank(&info.currentCoord[0],user);
                    VertexData vd = VertexData(VertexID(pr,rank,&manager.localVertexCounter),info);
                    vd.fd = fd;
                    add_vertex(vd,manager.graph);
              }

            }

          	user->nen  = 0;
          	user->nen  = user->iga->axis[0]->p + 1;
          	user->nen *= user->iga->axis[1]->p + 1;
          	user->nen *= user->iga->axis[2]->p + 1;
  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "ParticleDistribute"
PetscErrorCode ParticleDistribute(PARAMETERS *par, AppCtx *user, ParticleManager &manager)
{
  // Based on the information about the PD initialization, update ID, ID_PD and material info
  // so that other quantities in the iteration loop can be computed conditioned on this information.
  //Debugging
  //PetscPrintf(PETSC_COMM_SELF, "%d\n", user->numFluidNodes);

  PetscFunctionBegin;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  user->totalInitialExplosiveVolume = 0.0;
  user->totalCurrentExplosiveVolume = 0.0;
  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
  // Here we map to the PD object ID so that the kinematic fields can be updated
  for(auto it=its.first; it != its.second; ++it){
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    FieldData fd = get(fd_property,v);
    ParticleInfo info = get(info_property,v);
    fd.ID_PD = -1;
    // In Pre-Processor, index starts at 1, so we correct so that it is 0->N-1 so it can be used for
    // indexing arrays which correspond to i = 1 : N
  if(fd.material==0 && fd.ID>user->numFluidNodes && !info.isTask){
    fd.ID_PD = fd.ID-user->numFluidNodes-1;
    PetscReal x = info.currentCoord[0];
    PetscReal y = info.currentCoord[1];
    PetscReal z = info.currentCoord[2];

    if(fd.ID_PD<0){
    PetscPrintf(PETSC_COMM_SELF, "PD_ID < 0 : number of processors < number of input files \n");
    exit(0);
    }
  }

  if(fd.material==1){
  user->totalInitialExplosiveVolume+=fd.nodalVolume;
  user->totalCurrentExplosiveVolume+=fd.nodalVolume;
  }
  put(fd_property,v,fd);

}

  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "outputTXT"
PetscErrorCode outputTXT (PARAMETERS *par, ParticleManager &manager)
{


  PetscInt count=par->numNodes;
  PetscInt outCount=par->stepNumber;
  PetscInt counter=par->stepNumber/par->FreqResults;

  PetscInt num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ostringstream convert;
  ostringstream convert1;
  ostringstream convert2;
  ostringstream convert3;
  ostringstream convert4;
  ostringstream convert5;
  PetscInt i,j;
  count = 0;
  int count1 = 0;
  BGL_FORALL_VERTICES(v,manager.graph,Graph){
    FieldData fd = get(fd_property,v);
    ParticleInfo info = get(info_property,v);
    if(!info.isTask){
        count++;
        // Geo << count1 <<  "  " << scientific << setprecision(4) << info.currentCoord[0] << "  " << info.currentCoord[1] << "  " << info.currentCoord[2] <$
        }
    }

  if (count > 0){
  if (outCount % par->FreqResults == 0){

    // ##################################################
    //                  Geometry File
    // ##################################################

    convert << "Meshless." << rank << "." << counter << ".geo";
    string fname = convert.str();
    ofstream Geo;
    Geo.open(fname.c_str());

    Geo << "Meshless" << endl;
    Geo << "node" << endl;
    Geo << "node id given" << endl;
    Geo << "element id given" << endl;
    Geo << "coordinates" << endl;
    Geo << count << endl;

           BGL_FORALL_VERTICES(v,manager.graph,Graph){
           FieldData fd = get(fd_property,v);
           ParticleInfo info = get(info_property,v);
            if(!info.isTask){
                  count1++;
                  Geo << count1 <<  "  " << scientific << setprecision(4) << info.currentCoord[0] << "  " << info.currentCoord[1] << "  " << info.currentCoord[2] << endl;
            }
           }
    Geo << "part    1" << endl;
    Geo << "todo" << endl;
    Geo << "point" << endl;
    Geo << count << endl;
      for(j=1;j<=count;j++){
         Geo << j << " " << j << endl;
      }
      Geo.close();


      // ##################################################
      //                  Case File
      // ##################################################

      if (outCount == 0){
        convert1 << "Meshfree." << rank << ".case";
        fname = convert1.str();
        ofstream Case;
        Case.open(fname.c_str());

      PetscInt numSteps = (int) (par->finalTime/par->timeStep);

      Case << "#BOF: meshless.case" << endl;
      Case << endl;
      Case << "FORMAT" << endl;
      Case << endl;
      Case << "type: ensight" << endl;
      Case << endl;
      Case << "GEOMETRY" << endl;
      Case << endl;
      Case << "model: 1 Meshless." << rank << ".*.geo" << endl;
      Case << endl;
      Case << "VARIABLE" << endl;
      Case << endl;
      Case << "scalar per node: 1 Density Density." << rank << ".*.res" << endl;
      Case << "vector per node: 1 Velocity Velocity." << rank << ".*.res" << endl;
      Case << "scalar per node: 1 Pressure Pressure." << rank << ".*.res" << endl;
      Case << endl;
      Case << "TIME" << endl;
      Case << endl;
      Case << "time set: 1" << endl;
      Case << "number of steps: " << numSteps/par->FreqResults << endl;
      Case << "filename start number: 0" << endl;
      Case << "filename increment: " << "1" << endl;
      Case << "time values:" << endl;

      PetscInt counter1=0;
      for(i=0; i< (par->finalTime/par->timeStep);i++){
        if (i % par->FreqResults ==0){
          Case << counter1 << " ";
          counter1++;
        }
        if ((i+1) % (10*par->FreqResults) == 0) Case << endl;
      }
      Case.close();
      }


//  // ##################################################
//  //                  Density File
//  // ##################################################

      convert2 << "Density." << rank << "." << counter << ".res";
      fname = convert2.str();
      ofstream Density;
      Density.open(fname.c_str());

      Density << "Density" << endl;
      PetscInt counter2 = 0;
           BGL_FORALL_VERTICES(v,manager.graph,Graph){
             ParticleInfo info = get(info_property,v);
             FieldData fd = get(fd_property,v);
            if(!info.isTask){
              Density << scientific << setprecision(4) << fd.nodalDensity << "  ";
              counter2++;
              if (((counter2) % 6 == 0)) Density << endl;
            }
           }
           Density.close();

convert4 << "Pressure." << rank << "." << counter << ".res";
fname = convert4.str();
ofstream Pressure;
Pressure.open(fname.c_str());
Pressure << "Pressure" << endl;
counter2 = 0;
BGL_FORALL_VERTICES(v,manager.graph,Graph){
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);
    if(!info.isTask){
      Pressure << scientific << setprecision(4) << fd.nodalPressure << "  ";
      counter2++;
      if (((counter2) % 6 == 0)) Pressure << endl;
        }
      }
    Pressure.close();

//  // ##################################################
//  //                  Velocity File
//  // ##################################################

          convert5 << "Velocity." << rank << "." << counter << ".res";
          fname = convert5.str();
          ofstream Velocity;
          Velocity.open(fname.c_str());
          PetscInt counter3 = 0;
          Velocity << "Velocity" << endl;
               BGL_FORALL_VERTICES(v,manager.graph,Graph){
                 ParticleInfo info = get(info_property,v);
                 FieldData fd = get(fd_property,v);
                if(!info.isTask){
                  Velocity << scientific << setprecision(4) << fd.totalPhysicalVelocity[0] << "  " << fd.totalPhysicalVelocity[1] << "  " << fd.totalPhysicalVelocity[2] << "  "  ;
                  counter3++;
                  if (((counter3) % 2 == 0)) Velocity << endl;
                }

               }
               Velocity.close();

  }
}
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "OutputRestarts"
PetscErrorCode OutputRestarts(PARAMETERS *par,Vec U,Vec V,ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt i,j;
  PetscInt count=par->numNodes;


  PetscInt num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ostringstream convert;
  ostringstream convert1;
  ostringstream convert2;
  ostringstream convert3;
  ostringstream convert4;
  ostringstream convert5;
  ostringstream convert6;
  ostringstream convert7;
  ostringstream convert8;
  ostringstream convert9;
  ostringstream convert10;
  ostringstream convert11;

  string fname;

  PetscInt counter = 0; //Number of nodes stored in current rank
  BGL_FORALL_VERTICES(v,manager.graph,Graph){
 	 ParticleInfo info = get(info_property,v);
 	 if(!info.isTask){
 		 counter++;
 	 }
  }

  // ##################################################
  //                  Velocity File
  // ##################################################

    MPI_Comm comm;
    PetscViewer viewer;
    ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
    char filenameRestart[256];
    sprintf(filenameRestart,"RestartU%d.dat",par->stepNumber);
    ierr = PetscViewerBinaryOpen(comm,filenameRestart,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(U,viewer);CHKERRQ(ierr);

    ierr = PetscObjectGetComm((PetscObject)V,&comm);CHKERRQ(ierr);

    sprintf(filenameRestart,"RestartV%d.dat",par->stepNumber);
    ierr = PetscViewerBinaryOpen(comm,filenameRestart,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = VecView(V,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);

		convert5 << "RestartVelocity." << rank << "." << par->stepNumber << ".dat";
		fname = convert5.str();
		ofstream Velocity;
		Velocity.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Acceleration File
		// ##################################################

		  convert4 << "RestartAcceleration." << rank << "." << par->stepNumber << ".dat";
		  fname = convert4.str();
		  ofstream Acceleration;
		  Acceleration.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Stress File
		// ##################################################

		convert3 << "RestartStress." << rank << "." << par->stepNumber << ".dat";
		fname = convert3.str();
		ofstream Stress;
		Stress.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Strain File
		// ##################################################

		convert2 << "RestartStrain." << rank << "." << par->stepNumber << ".dat";
		fname = convert2.str();
		ofstream Strain;
		Strain.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Geometry File
		// ##################################################

		convert1 << "RestartGeo." << rank << "." << par->stepNumber << ".dat";
		fname = convert1.str();
		ofstream Geo;
		Geo.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Damage File
		// ##################################################

		convert << "RestartDamage." << rank << "." << par->stepNumber << ".dat";
		fname = convert.str();
		ofstream Damage;
		Damage.open(fname, ios::out | ios::binary);

		// ##################################################
		//               Damage Threshold File
		// ##################################################

		convert6 << "RestartThreshold." << rank << "." << par->stepNumber << ".dat";
		fname = convert6.str();
		ofstream Threshold;
		Threshold.open(fname, ios::out | ios::binary);

		// ##################################################
		//             Deformation Gradient File
		// ##################################################

		convert7 << "RestartDefGrad." << rank << "." << par->stepNumber << ".dat";
		fname = convert7.str();
		ofstream DefGrad;
		DefGrad.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Material File
		// ##################################################

		convert8 << "RestartMat." << rank << "." << par->stepNumber << ".dat";
		fname = convert8.str();
		ofstream Mat;
		Mat.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Volume File
		// ##################################################

		convert9 << "RestartVol." << rank << "." << par->stepNumber << ".dat";
		fname = convert9.str();
		ofstream Vol;
		Vol.open(fname, ios::out | ios::binary);

		// ##################################################
		//                  Boundary File
		// ##################################################

		convert10 << "RestartBound." << rank << "." << par->stepNumber << ".dat";
		fname = convert10.str();
		ofstream Bound;
		Bound.open(fname, ios::out | ios::binary);

    // ##################################################
    //              Number of particles File
    // ##################################################

    convert11 << "RestartNum." << rank << "." << par->stepNumber << ".dat";
    fname = convert11.str();
    ofstream Num;
    Num.open(fname, ios::out | ios::binary);
		Num.write( (char*)&counter, sizeof(int));


		if (counter > 0){
		   BGL_FORALL_VERTICES(v,manager.graph,Graph){
			 ParticleInfo info = get(info_property,v);
			 FieldData fd = get(fd_property,v);
			if(!info.isTask){

				Velocity.write( (char*)&fd.totalPhysicalVelocity[0], sizeof(double));
				Velocity.write( (char*)&fd.totalPhysicalVelocity[1], sizeof(double));
				Velocity.write( (char*)&fd.totalPhysicalVelocity[2], sizeof(double));
        Velocity.write( (char*)&fd.interpolatedVelocity[0], sizeof(double));
				Velocity.write( (char*)&fd.interpolatedVelocity[1], sizeof(double));
				Velocity.write( (char*)&fd.interpolatedVelocity[2], sizeof(double));

				Acceleration.write( (char*)&fd.totalPhysicalAcceleration[0], sizeof(double));
				Acceleration.write( (char*)&fd.totalPhysicalAcceleration[1], sizeof(double));
				Acceleration.write( (char*)&fd.totalPhysicalAcceleration[2], sizeof(double));

				Stress.write( (char*)&fd.totalStress0[0], sizeof(double));
				Stress.write( (char*)&fd.totalStress0[1], sizeof(double));
				Stress.write( (char*)&fd.totalStress0[2], sizeof(double));
				Stress.write( (char*)&fd.totalStress0[3], sizeof(double));
				Stress.write( (char*)&fd.totalStress0[4], sizeof(double));
				Stress.write( (char*)&fd.totalStress0[5], sizeof(double));

				Strain.write( (char*)&fd.totalStrain0[0], sizeof(double));
				Strain.write( (char*)&fd.totalStrain0[1], sizeof(double));
				Strain.write( (char*)&fd.totalStrain0[2], sizeof(double));
				Strain.write( (char*)&fd.totalStrain0[3], sizeof(double));
				Strain.write( (char*)&fd.totalStrain0[4], sizeof(double));
				Strain.write( (char*)&fd.totalStrain0[5], sizeof(double));

        Geo.write( (char*)&info.initialCoord[0], sizeof(double));
				Geo.write( (char*)&info.initialCoord[1], sizeof(double));
				Geo.write( (char*)&info.initialCoord[2], sizeof(double));
				Geo.write( (char*)&info.currentCoord[0], sizeof(double));
				Geo.write( (char*)&info.currentCoord[1], sizeof(double));
				Geo.write( (char*)&info.currentCoord[2], sizeof(double));
        Geo.write( (char*)&fd.computed_currentCoord[0], sizeof(double));
        Geo.write( (char*)&fd.computed_currentCoord[1], sizeof(double));
        Geo.write( (char*)&fd.computed_currentCoord[2], sizeof(double));
        Geo.write( (char*)&fd.totalPhysicalDisplacement[0], sizeof(double));
        Geo.write( (char*)&fd.totalPhysicalDisplacement[1], sizeof(double));
        Geo.write( (char*)&fd.totalPhysicalDisplacement[2], sizeof(double));

				DefGrad.write( (char*)&fd.currentDeformationGradient[0], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[1], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[2], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[3], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[4], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[5], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[6], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[7], sizeof(double));
				DefGrad.write( (char*)&fd.currentDeformationGradient[8], sizeof(double));

				Damage.write( (char*)&fd.damage0, sizeof(double));
        Damage.write( (char*)&fd.damage, sizeof(double));

				Threshold.write( (char*)&fd.ductile_threshold0, sizeof(double));
				Threshold.write( (char*)&fd.brittle_threshold0, sizeof(double));

				Mat.write( (char*)&fd.material, sizeof(int));
        Mat.write( (char*)&fd.ID_PD, sizeof(int));
        Mat.write( (char*)&fd.ID, sizeof(int));

        //Volume file includes information for volume update
        //[volume, volume_initial, nodalDensity, nodalDensityInitial]
				Vol.write( (char*)&fd.nodalVolume, sizeof(double));
        Vol.write( (char*)&fd.nodalVolumeInitial, sizeof(double));
        Vol.write( (char*)&fd.nodalDensity, sizeof(double));
        Vol.write( (char*)&fd.nodalDensityInitial, sizeof(double));
        Vol.write( (char*)&fd.penaltyParameter, sizeof(double));
        Vol.write( (char*)&fd.referencePenaltyParameterInternal, sizeof(double));

				Bound.write( (char*)&fd.Boundary, sizeof(int));
        Bound.write( (char*)&fd.Inside, sizeof(int));
			}
		   }
    }
		   Velocity.close();
		   Acceleration.close();
		   Stress.close();
		   Strain.close();
		   Geo.close();
		   Damage.close();
		   Threshold.close();
		   DefGrad.close();
		   Mat.close();
		   Vol.close();
		   Bound.close();
		   Num.close();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadLastResults"
PetscErrorCode ReadLastResults(PARAMETERS *par,Vec U,Vec V,PetscInt StepRestart, PetscInt FreqRestarts,ParticleManager &manager,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscInt i,j;
  PetscInt num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  MPI_Comm comm;
  PetscViewer viewer;

  // U
  char filename[256];
  sprintf(filename,"RestartU%d.dat",StepRestart);
  ierr = PetscObjectGetComm((PetscObject)U,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(U,viewer);CHKERRQ(ierr);

  //V
  sprintf(filename,"RestartV%d.dat",StepRestart);
  ierr = PetscObjectGetComm((PetscObject)V,&comm);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(comm,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(V,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);

  // TotalStress
  ostringstream convert;
  convert << "RestartStress." << rank << "." << StepRestart << ".dat";
  string fname = convert.str();
  ifstream Stress;
  Stress.open(fname, ios::in | ios::binary);


    // TotalStrain
    ostringstream convert1;
    convert1 << "RestartStrain." << rank << "." << StepRestart << ".dat";
    fname = convert1.str();
    ifstream Strain;
    Strain.open(fname, ios::in | ios::binary);

    // Geometry
    ostringstream convert2;
    convert2 << "RestartGeo." << rank << "." << StepRestart << ".dat";
    fname = convert2.str();
    ifstream Geo;
    Geo.open(fname, ios::in | ios::binary);

    // Velocity
    ostringstream convert4;
    convert4 << "RestartVelocity." << rank << "." << StepRestart << ".dat";
    fname = convert4.str();
    ifstream Velocity;
    Velocity.open(fname, ios::in | ios::binary);

    // Acceleration
    ostringstream convert5;
    convert5 << "RestartAcceleration." << rank << "." << StepRestart << ".dat";
    fname = convert5.str();
    ifstream Acceleration;
    Acceleration.open(fname, ios::in | ios::binary);

    // Damage
    ostringstream convert6;
    convert6 << "RestartDamage." << rank << "." << StepRestart << ".dat";
    fname = convert6.str();
    ifstream Damage;
    Damage.open(fname, ios::in | ios::binary);

    // Damage Thresholds
    ostringstream convert7;
    convert7 << "RestartThreshold." << rank << "." << StepRestart << ".dat";
    fname = convert7.str();
    ifstream Threshold;
    Threshold.open(fname, ios::in | ios::binary);

    //Material
    ostringstream convert9;
    convert9 << "RestartMat." << rank << "." << StepRestart << ".dat";
    fname = convert9.str();
    ifstream Mat;
    Mat.open(fname, ios::in | ios::binary);

    // Volume
    ostringstream convert10;
    convert10 << "RestartVol." << rank << "." << StepRestart << ".dat";
    fname = convert10.str();
    ifstream Vol;
    Vol.open(fname, ios::in | ios::binary);

    // Boundary
    ostringstream convert11;
    convert11 << "RestartBound." << rank << "." << StepRestart << ".dat";
    fname = convert11.str();
    ifstream Bound;
    Bound.open(fname, ios::in | ios::binary);

    // Number of particles
    ostringstream convert12;
    convert12 << "RestartNum." << rank << "." << StepRestart << ".dat";
    fname = convert12.str();
    ifstream Num;
    Num.open(fname, ios::in | ios::binary);

    // Deformation Gradient
    ostringstream convert13;
    convert13 << "RestartDefGrad." << rank << "." << StepRestart << ".dat";
    fname = convert13.str();
    ifstream DefGrad;
    DefGrad.open(fname, ios::in | ios::binary);


    PetscInt counter; //Number of particles on current rank
    Num.read( (char*)&counter, sizeof(int));
    par->numNodes = counter;

    user->totalInitialExplosiveVolume = 0.0;
    user->totalCurrentExplosiveVolume = 0.0;

    if (counter > 0){
    for (i = 0; i < counter; i++){

        ParticleInfo info = ParticleInfo();
        FieldData fd = FieldData();


		Stress.read( (char*)&fd.totalStress0[0], sizeof(double));
		Stress.read( (char*)&fd.totalStress0[1], sizeof(double));
		Stress.read( (char*)&fd.totalStress0[2], sizeof(double));
		Stress.read( (char*)&fd.totalStress0[3], sizeof(double));
		Stress.read( (char*)&fd.totalStress0[4], sizeof(double));
		Stress.read( (char*)&fd.totalStress0[5], sizeof(double));

 	  Strain.read( (char*)&fd.totalStrain0[0], sizeof(double));
	  Strain.read( (char*)&fd.totalStrain0[1], sizeof(double));
	  Strain.read( (char*)&fd.totalStrain0[2], sizeof(double));
	  Strain.read( (char*)&fd.totalStrain0[3], sizeof(double));
	  Strain.read( (char*)&fd.totalStrain0[4], sizeof(double));
	  Strain.read( (char*)&fd.totalStrain0[5], sizeof(double));

	  DefGrad.read( (char*)&fd.currentDeformationGradient[0], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[1], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[2], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[3], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[4], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[5], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[6], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[7], sizeof(double));
	  DefGrad.read( (char*)&fd.currentDeformationGradient[8], sizeof(double));

    Geo.read( (char*)&info.initialCoord[0], sizeof(double));
	  Geo.read( (char*)&info.initialCoord[1], sizeof(double));
 	  Geo.read( (char*)&info.initialCoord[2], sizeof(double));
    Geo.read( (char*)&info.currentCoord[0], sizeof(double));
	  Geo.read( (char*)&info.currentCoord[1], sizeof(double));
 	  Geo.read( (char*)&info.currentCoord[2], sizeof(double));
    Geo.read( (char*)&fd.computed_currentCoord[0], sizeof(double));
    Geo.read( (char*)&fd.computed_currentCoord[1], sizeof(double));
    Geo.read( (char*)&fd.computed_currentCoord[2], sizeof(double));
    Geo.read( (char*)&fd.totalPhysicalDisplacement[0], sizeof(double));
    Geo.read( (char*)&fd.totalPhysicalDisplacement[1], sizeof(double));
    Geo.read( (char*)&fd.totalPhysicalDisplacement[2], sizeof(double));

     Velocity.read( (char*)&fd.totalPhysicalVelocity[0], sizeof(double));
     Velocity.read( (char*)&fd.totalPhysicalVelocity[1], sizeof(double));
     Velocity.read( (char*)&fd.totalPhysicalVelocity[2], sizeof(double));
     Velocity.read( (char*)&fd.interpolatedVelocity[0], sizeof(double));
     Velocity.read( (char*)&fd.interpolatedVelocity[1], sizeof(double));
     Velocity.read( (char*)&fd.interpolatedVelocity[2], sizeof(double));

     Acceleration.read( (char*)&fd.totalPhysicalAcceleration[0], sizeof(double));
     Acceleration.read( (char*)&fd.totalPhysicalAcceleration[1], sizeof(double));
     Acceleration.read( (char*)&fd.totalPhysicalAcceleration[2], sizeof(double));

     Damage.read( (char*)&fd.damage0, sizeof(double));
     Damage.read( (char*)&fd.damage, sizeof(double));

     Threshold.read( (char*)&fd.ductile_threshold0, sizeof(double));
	   Threshold.read( (char*)&fd.brittle_threshold0, sizeof(double));

     Mat.read( (char*)&fd.material, sizeof(int));
     Mat.read( (char*)&fd.ID_PD, sizeof(int));
     Mat.read( (char*)&fd.ID, sizeof(int));

     Vol.read( (char*)&fd.nodalVolume, sizeof(double));
     Vol.read( (char*)&fd.nodalVolumeInitial, sizeof(double));
     Vol.read( (char*)&fd.nodalDensity, sizeof(double));
     Vol.read( (char*)&fd.nodalDensityInitial, sizeof(double));
     Vol.read( (char*)&fd.penaltyParameter, sizeof(double));
     Vol.read( (char*)& fd.referencePenaltyParameterInternal, sizeof(double));

     if (fd.material == 1){
        user->totalInitialExplosiveVolume += fd.nodalVolumeInitial;
        user->totalCurrentExplosiveVolume += fd.nodalVolume;
      }
     Bound.read( (char*)&fd.Boundary, sizeof(int));
     Bound.read( (char*)&fd.Inside, sizeof(int));

		VertexData vd = VertexData(VertexID(rank,rank,&manager.localVertexCounter),info);
		vd.fd = fd;
		add_vertex(vd,manager.graph);
        }

  }

  Stress.close();
  Strain.close();
  Geo.close();
  Velocity.close();
  Acceleration.close();
  Damage.close();
  Threshold.close();
  Mat.close();
  Vol.close();
  Bound.close();
  Num.close();
  DefGrad.close();

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "GetElementInfo"
PetscErrorCode GetElementInfo(PARAMETERS *par,ParticleManager &manager, AppCtx *user)
{
  PetscErrorCode ierr = 0;
  ostringstream ElemInfo;
  ElemInfo << "ElementInfo" << ".dat";
  string ElemInfoName = ElemInfo.str();
  ifstream ELin;
  ELin.open(ElemInfoName.c_str());

  PetscMalloc1(user->iga->axis[0]->nel+1, &user->xhat);
  PetscMalloc1(user->iga->axis[1]->nel+1, &user->yhat);
  PetscMalloc1(user->iga->axis[2]->nel+1, &user->zhat);
  //PetscPrintf(PETSC_COMM_WORLD, "%d %d %d \n unique knot points read in each direction \n", user->iga->axis[0]->nel+1, user->iga->axis[1]->nel+1, user->iga->axis[2]->nel+1);
  for(int i = 0; i<(user->iga->axis[0]->nel+1); i++){
    ELin >> user->xhat[i];
    //PetscPrintf(PETSC_COMM_WORLD, "X  = %e \n", user->xhat[i]);
  }
  for(int i = 0; i<(user->iga->axis[1]->nel+1); i++){
    ELin >> user->yhat[i];
    //PetscPrintf(PETSC_COMM_WORLD, "Y = %e \n", user->yhat[i]);
  }
  for(int i = 0; i<(user->iga->axis[2]->nel+1); i++){
    ELin >> user->zhat[i];
    //PetscPrintf(PETSC_COMM_WORLD, "Z = %e \n", user->zhat[i]);
  }

PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialCondition"
PetscErrorCode FormInitialCondition(IGA iga,PetscReal t,Vec U,AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  DM da;
  PetscInt dof = iga->dof;
  PetscInt dim = iga->dim;

  ierr = IGACreateNodeDM(iga,dof,&da);CHKERRQ(ierr);
  Field ***u;
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  DMDALocalInfo info;
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);


  PetscInt i,j,k;
  PetscInt nodesX  = iga->geom_lwidth[0], nodesY  = iga->geom_lwidth[1], nodesZ  = iga->geom_lwidth[2];
  PetscInt gnodesX = iga->geom_gwidth[0], gnodesY = iga->geom_gwidth[1];


  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      for(k=info.zs;k<info.zs+info.zm;k++){

        PetscReal x = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim];
        PetscReal y = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 1];
        PetscReal z = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 2];

        PetscReal pt[3];
        pt[0] = x/user->Lx;
        pt[1] = y/user->Ly;
        pt[2] = z/user->Lz;

        PetscReal h = IGAGetElementDiagonal(iga, pt, user);
        h/=sqrt(3.0); //Obtain side length of element (again, initialize in unif region)
        PetscReal theta = atan2(y,x);
        PetscReal r     = sqrt(x*x+y*y);
        PetscReal phi   = 0;
        PetscReal height= 0.5*0.005;
        PetscReal R     = 0.00472605997;
        PetscReal factor= h + sin(2*theta)*(sqrt(2.0)*h-h);

        //Shallow water Tank conditions
        u[k][j][i].ux   =  0.0;
        u[k][j][i].uz   =  0.0;
        u[k][j][i].uy   =  0.0;
        u[k][j][i].rho  =  1000.0;
        u[k][j][i].temp =  290.0;
        u[k][j][i].alpha = 0.0;

        //RDX Charge: must initialize in uniform region
        if(r <= R + factor && z <= height + h && r < 1.5*R){
          //u[k][j][i].alpha = 1.0;
          u[k][j][i].rho   = 1770.0;
        }
        //We want the 0.5 level set to be coincedent with the boundary of the constant region,
        //so we actually do not want the constant region to be of constant alpha
        if(r <= R && z <= height){
          u[k][j][i].alpha = 1.0;
          //u[k][j][i].rho   = 1770.0;
        }

    }
  }
}
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

//// End I/O ////



//// Stabilization Parameters ////

////////////////////////////////// 5 DOF ////////////////////////////////
#undef __FUNCT__
#define __FUNCT__ "Invert5x5_eigen"
PetscErrorCode Invert5x5_eigen( PetscScalar (*A)[5],PetscReal (*Ainv)[5],void *ctx)
{

    AppCtx *user = (AppCtx *)ctx;

    PetscReal DET;
    PetscInt  i,j;

    double data[5][5];
    for(i = 0 ; i <  5 ; i++){
      for(j = 0 ; j <  5 ; j++){
        data[i][j] = A[i][j];
      }
    }

    Eigen::Map<Matrix<double,5,5,RowMajor> > mat(data[0]);

    mat = mat.inverse();

    for(i = 0 ; i <  5 ; i++){
      for(j = 0 ; j <  5 ; j++){
        Ainv[i][j] = mat(i*5+j);
      }
    }


    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Compute_square_root_inverse"
PetscErrorCode Compute_square_root_inverse(PetscReal (*TauTemp)[5],PetscReal (*TauTempInv)[5],PetscReal taut,void *ctx)
{

    AppCtx *user = (AppCtx *)ctx;
    PetscInt  dof    = user->iga->dof - user->numLS_Fields;

    PetscInt  i,j,aa;
    PetscReal Y[5][5]={{0.0}};
    PetscReal Z[5][5]={{0.0}};
    PetscReal Yinv[5][5]={{0.0}};
    PetscReal Zinv[5][5]={{0.0}};

    for (i=0;i<dof;i++){
      Z[i][i] = 1.0;
      for (j=0;j<dof;j++){
            Y[i][j]= TauTemp[i][j]/taut;
      }
    }

    for (aa=0;aa<10;aa++){
        Invert5x5_eigen(Y, Yinv, user);
        Invert5x5_eigen(Z, Zinv, user);

        for (i=0;i<dof;i++){
            for (j=0;j<dof;j++){
                Y[i][j]= 0.5*(Y[i][j] + Zinv[i][j]);
            }
        }

        for (i=0;i<dof;i++){
            for (j=0;j<dof;j++){
                Z[i][j]= 0.5*(Z[i][j] + Yinv[i][j]);
            }
        }
      }

    for (i=0;i<dof;i++){
        for (j=0;j<dof;j++){
            TauTempInv[i][j]= Z[i][j]/sqrt(taut);
        }
    }

    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeAMatrixConservation"
PetscErrorCode ComputeAMatrixConservation(PetscScalar u[],
                                          PetscReal (*A0inv)[6],PetscReal (*A1_c)[6],PetscReal (*A1_p)[6], PetscReal (*A1_pt)[6],
                                          PetscReal (*A2_c)[6],PetscReal (*A2_p)[6], PetscReal (*A2_pt)[6], PetscReal (*A3_c)[6],
                                          PetscReal (*A3_p)[6], PetscReal (*A3_pt)[6], PetscReal (*K)[3][4][4],
                                          PetscReal (*A1_cons)[5],PetscReal (*A2_cons)[5],PetscReal (*A3_cons)[5],PetscReal (*K_cons)[3][5][5],void *ctx)
{
    AppCtx *user = (AppCtx *)ctx;
    PetscInt  dof    = user->iga->dof - user->numLS_Fields;
    PetscInt i,j,l;

    for (i=0;i<dof;i++){
        for (j=0;j<dof;j++){
            for (l=0;l<dof;l++){
                A1_cons[i][j] += (A1_p[i][l]+A1_c[i][l]+A1_pt[i][l])*A0inv[l][j];
                A2_cons[i][j] += (A2_p[i][l]+A2_c[i][l]+A2_pt[i][l])*A0inv[l][j];
                A3_cons[i][j] += (A3_p[i][l]+A3_c[i][l]+A3_pt[i][l])*A0inv[l][j];
            }
        }
    }

    for (i=0;i<dof-1;i++){
        for (j=0;j<dof;j++){
            for (l=0;l<dof-1;l++){
                K_cons[0][0][i+1][j] += K[0][0][i][l]*A0inv[l+1][j];
                K_cons[0][1][i+1][j] += K[0][1][i][l]*A0inv[l+1][j];
                K_cons[0][2][i+1][j] += K[0][2][i][l]*A0inv[l+1][j];
                K_cons[1][0][i+1][j] += K[1][0][i][l]*A0inv[l+1][j];
                K_cons[1][1][i+1][j] += K[1][1][i][l]*A0inv[l+1][j];
                K_cons[1][2][i+1][j] += K[1][2][i][l]*A0inv[l+1][j];
                K_cons[2][0][i+1][j] += K[2][0][i][l]*A0inv[l+1][j];
                K_cons[2][1][i+1][j] += K[2][1][i][l]*A0inv[l+1][j];
                K_cons[2][2][i+1][j] += K[2][2][i][l]*A0inv[l+1][j];
            }
        }
    }

    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "DirectTau"
PetscErrorCode DirectTau(PetscReal G[3][3],
                         PetscReal dt,PetscScalar u[],
                         PetscScalar (*tau)[5],void *ctx,PetscReal (*A0inv)[6],PetscReal (*A1_cons)[5],PetscReal (*A2_cons)[5],PetscReal (*A3_cons)[5],PetscReal (*K_cons)[3][5][5],PetscReal *umi)
{

    AppCtx *user = (AppCtx *)ctx;

    PetscInt  dof    = user->iga->dof - user->numLS_Fields;


    //Tau for Conservation variables

    PetscReal TauTemp[5][5]={{0.0}};
    PetscReal TauTempInv[5][5]={{0.0}};
    PetscReal taut=0.0;

    taut = 4.0/(dt*dt);

    PetscInt aa,bb,ll;
    for (aa=0;aa<dof;aa++)
        TauTemp[aa][aa] = taut;



    for (aa=0;aa<dof;aa++)
        for (bb=0;bb<dof;bb++)
            for (ll=0;ll<dof;ll++){
                TauTemp[aa][bb] += G[0][0]*A1_cons[aa][ll]*A1_cons[ll][bb]
                +  G[0][1]*A1_cons[aa][ll]*A2_cons[ll][bb]
                +  G[0][2]*A1_cons[aa][ll]*A3_cons[ll][bb]
                +  G[1][0]*A2_cons[aa][ll]*A1_cons[ll][bb]
                +  G[1][1]*A2_cons[aa][ll]*A2_cons[ll][bb]
                +  G[1][2]*A2_cons[aa][ll]*A3_cons[ll][bb]
                +  G[2][0]*A3_cons[aa][ll]*A1_cons[ll][bb]
                +  G[2][1]*A3_cons[aa][ll]*A2_cons[ll][bb]
                +  G[2][2]*A3_cons[aa][ll]*A3_cons[ll][bb];
            }


    PetscReal m_k=3.0;

    for (aa=0;aa<dof;aa++)
        for (bb=0;bb<dof;bb++)
  	  	  	   for (ll=0;ll<dof;ll++){


                   TauTemp[aa][bb] += m_k*G[0][0]*K_cons[0][0][aa][ll]*K_cons[0][0][ll][bb]*G[0][0]
                   +  m_k*G[0][0]*K_cons[0][1][aa][ll]*K_cons[0][0][ll][bb]*G[0][1]
                   +  m_k*G[0][0]*K_cons[0][2][aa][ll]*K_cons[0][0][ll][bb]*G[0][2]

                   +  m_k*G[0][1]*K_cons[1][0][aa][ll]*K_cons[0][0][ll][bb]*G[0][0]
                   +  m_k*G[0][1]*K_cons[1][1][aa][ll]*K_cons[0][0][ll][bb]*G[0][1]
                   +  m_k*G[0][1]*K_cons[1][2][aa][ll]*K_cons[0][0][ll][bb]*G[0][2]

                   +  m_k*G[0][2]*K_cons[2][0][aa][ll]*K_cons[0][0][ll][bb]*G[0][0]
                   +  m_k*G[0][2]*K_cons[2][1][aa][ll]*K_cons[0][0][ll][bb]*G[0][1]
                   +  m_k*G[0][2]*K_cons[2][2][aa][ll]*K_cons[0][0][ll][bb]*G[0][2]



                   +  m_k*G[0][0]*K_cons[0][0][aa][ll]*K_cons[0][1][ll][bb]*G[1][0]
                   +  m_k*G[0][0]*K_cons[0][1][aa][ll]*K_cons[0][1][ll][bb]*G[1][1]
                   +  m_k*G[0][0]*K_cons[0][2][aa][ll]*K_cons[0][1][ll][bb]*G[1][2]

                   +  m_k*G[0][1]*K_cons[1][0][aa][ll]*K_cons[0][1][ll][bb]*G[1][0]
                   +  m_k*G[0][1]*K_cons[1][1][aa][ll]*K_cons[0][1][ll][bb]*G[1][1]
                   +  m_k*G[0][1]*K_cons[1][2][aa][ll]*K_cons[0][1][ll][bb]*G[1][2]

                   +  m_k*G[0][2]*K_cons[2][0][aa][ll]*K_cons[0][1][ll][bb]*G[1][0]
                   +  m_k*G[0][2]*K_cons[2][1][aa][ll]*K_cons[0][1][ll][bb]*G[1][1]
                   +  m_k*G[0][2]*K_cons[2][2][aa][ll]*K_cons[0][1][ll][bb]*G[1][2]


                   +  m_k*G[0][0]*K_cons[0][0][aa][ll]*K_cons[0][2][ll][bb]*G[2][0]
                   +  m_k*G[0][0]*K_cons[0][1][aa][ll]*K_cons[0][2][ll][bb]*G[2][1]
                   +  m_k*G[0][0]*K_cons[0][2][aa][ll]*K_cons[0][2][ll][bb]*G[2][2]

                   +  m_k*G[0][1]*K_cons[1][0][aa][ll]*K_cons[0][2][ll][bb]*G[2][0]
                   +  m_k*G[0][1]*K_cons[1][1][aa][ll]*K_cons[0][2][ll][bb]*G[2][1]
                   +  m_k*G[0][1]*K_cons[1][2][aa][ll]*K_cons[0][2][ll][bb]*G[2][2]

                   +  m_k*G[0][2]*K_cons[2][0][aa][ll]*K_cons[0][2][ll][bb]*G[2][0]
                   +  m_k*G[0][2]*K_cons[2][1][aa][ll]*K_cons[0][2][ll][bb]*G[2][1]
                   +  m_k*G[0][2]*K_cons[2][2][aa][ll]*K_cons[0][2][ll][bb]*G[2][2]


                   +  m_k*G[1][0]*K_cons[0][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[1][0]*K_cons[0][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[1][0]*K_cons[0][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[1][1]*K_cons[1][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[1][1]*K_cons[1][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[1][1]*K_cons[1][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[1][2]*K_cons[2][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[1][2]*K_cons[2][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[1][2]*K_cons[2][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]


                   +  m_k*G[1][0]*K_cons[0][0][aa][ll]*K_cons[1][1][ll][bb]*G[1][0]
                   +  m_k*G[1][0]*K_cons[0][1][aa][ll]*K_cons[1][1][ll][bb]*G[1][1]
                   +  m_k*G[1][0]*K_cons[0][2][aa][ll]*K_cons[1][1][ll][bb]*G[1][2]

                   +  m_k*G[1][1]*K_cons[1][0][aa][ll]*K_cons[1][1][ll][bb]*G[1][0]
                   +  m_k*G[1][1]*K_cons[1][1][aa][ll]*K_cons[1][1][ll][bb]*G[1][1]
                   +  m_k*G[1][1]*K_cons[1][2][aa][ll]*K_cons[1][1][ll][bb]*G[1][2]

                   +  m_k*G[1][2]*K_cons[2][0][aa][ll]*K_cons[1][1][ll][bb]*G[1][0]
                   +  m_k*G[1][2]*K_cons[2][1][aa][ll]*K_cons[1][1][ll][bb]*G[1][1]
                   +  m_k*G[1][2]*K_cons[2][2][aa][ll]*K_cons[1][1][ll][bb]*G[1][2]


                   +  m_k*G[1][0]*K_cons[0][0][aa][ll]*K_cons[1][2][ll][bb]*G[2][0]
                   +  m_k*G[1][0]*K_cons[0][1][aa][ll]*K_cons[1][2][ll][bb]*G[2][1]
                   +  m_k*G[1][0]*K_cons[0][2][aa][ll]*K_cons[1][2][ll][bb]*G[2][2]

                   +  m_k*G[1][1]*K_cons[1][0][aa][ll]*K_cons[1][2][ll][bb]*G[2][0]
                   +  m_k*G[1][1]*K_cons[1][1][aa][ll]*K_cons[1][2][ll][bb]*G[2][1]
                   +  m_k*G[1][1]*K_cons[1][2][aa][ll]*K_cons[1][2][ll][bb]*G[2][2]

                   +  m_k*G[1][2]*K_cons[2][0][aa][ll]*K_cons[1][2][ll][bb]*G[2][0]
                   +  m_k*G[1][2]*K_cons[2][1][aa][ll]*K_cons[1][2][ll][bb]*G[2][1]
                   +  m_k*G[1][2]*K_cons[2][2][aa][ll]*K_cons[1][2][ll][bb]*G[2][2]



                   +  m_k*G[2][0]*K_cons[0][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[2][0]*K_cons[0][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[2][0]*K_cons[0][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]


                   +  m_k*G[2][1]*K_cons[1][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[2][1]*K_cons[1][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[2][1]*K_cons[1][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[2][2]*K_cons[2][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[2][2]*K_cons[2][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[2][2]*K_cons[2][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[2][0]*K_cons[0][0][aa][ll]*K_cons[2][1][ll][bb]*G[1][0]
                   +  m_k*G[2][0]*K_cons[0][1][aa][ll]*K_cons[2][1][ll][bb]*G[1][1]
                   +  m_k*G[2][0]*K_cons[0][2][aa][ll]*K_cons[2][1][ll][bb]*G[1][2]

                   +  m_k*G[2][1]*K_cons[1][0][aa][ll]*K_cons[2][1][ll][bb]*G[1][0]
                   +  m_k*G[2][1]*K_cons[1][1][aa][ll]*K_cons[2][1][ll][bb]*G[1][1]
                   +  m_k*G[2][1]*K_cons[1][2][aa][ll]*K_cons[2][1][ll][bb]*G[1][2]

                   +  m_k*G[2][2]*K_cons[2][0][aa][ll]*K_cons[2][1][ll][bb]*G[1][0]
                   +  m_k*G[2][2]*K_cons[2][1][aa][ll]*K_cons[2][1][ll][bb]*G[1][1]
                   +  m_k*G[2][2]*K_cons[2][2][aa][ll]*K_cons[2][1][ll][bb]*G[1][2]

                   +  m_k*G[2][0]*K_cons[0][0][aa][ll]*K_cons[2][2][ll][bb]*G[2][0]
                   +  m_k*G[2][0]*K_cons[0][1][aa][ll]*K_cons[2][2][ll][bb]*G[2][1]
                   +  m_k*G[2][0]*K_cons[0][2][aa][ll]*K_cons[2][2][ll][bb]*G[2][2]

                   +  m_k*G[2][1]*K_cons[1][0][aa][ll]*K_cons[2][2][ll][bb]*G[2][0]
                   +  m_k*G[2][1]*K_cons[1][1][aa][ll]*K_cons[2][2][ll][bb]*G[2][1]
                   +  m_k*G[2][1]*K_cons[1][2][aa][ll]*K_cons[2][2][ll][bb]*G[2][2]

                   +  m_k*G[2][2]*K_cons[2][0][aa][ll]*K_cons[2][2][ll][bb]*G[2][0]
                   +  m_k*G[2][2]*K_cons[2][1][aa][ll]*K_cons[2][2][ll][bb]*G[2][1]
                   +  m_k*G[2][2]*K_cons[2][2][aa][ll]*K_cons[2][2][ll][bb]*G[2][2];



               }



    Compute_square_root_inverse(TauTemp, TauTempInv, taut,user);

    //Transform to primitive variables
    for (aa=0;aa<dof;aa++)
        for (bb=0;bb<dof;bb++){
            tau[aa][bb] = A0inv[aa][0]*TauTempInv[0][bb]
            +A0inv[aa][1]*TauTempInv[1][bb]
            +A0inv[aa][2]*TauTempInv[2][bb]
            +A0inv[aa][3]*TauTempInv[3][bb]
            +A0inv[aa][4]*TauTempInv[4][bb];
        }
    //

    return 0;
}


////////////////////////////////// 6 DOF ////////////////////////////////
#undef __FUNCT__
#define __FUNCT__ "Invert6x6_eigen"
PetscErrorCode Invert6x6_eigen( PetscScalar (*A)[6],PetscReal (*Ainv)[6],void *ctx)
{

    AppCtx *user = (AppCtx *)ctx;
    PetscInt  i,j;

    double data[6][6];
    for(i = 0 ; i <  6 ; i++){
      for(j = 0 ; j <  6 ; j++){
        data[i][j] = A[i][j];
      }
    }

    Eigen::Map<Matrix<double,6,6,RowMajor> > mat(data[0]);
    mat = mat.inverse();

    for(i = 0 ; i <  6 ; i++){
      for(j = 0 ; j <  6 ; j++){
        Ainv[i][j] = mat(i*6+j);
      }
    }


    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Compute_square_root_inverse_6x6"
PetscErrorCode Compute_square_root_inverse_6x6(PetscReal (*TauTemp)[6],PetscReal (*TauTempInv)[6],PetscReal taut,void *ctx)
{
    //Compute square root inverse of matrix tauTemp and put in TauTempInv using Eigen for SUPG operator for systems with 6 DOF
    AppCtx *user = (AppCtx *)ctx;
    PetscInt dof = user->iga->dof;

    PetscInt  i,j,aa;
    PetscReal Y[6][6]={{0.0}};
    PetscReal Z[6][6]={{0.0}};
    PetscReal Yinv[6][6]={{0.0}};
    PetscReal Zinv[6][6]={{0.0}};

    for (i=0;i<dof;i++){
      Z[i][i] = 1.0;
      for (j=0;j<dof;j++){
            Y[i][j]= TauTemp[i][j]/taut;
      }
    }

    for (aa=0;aa<10;aa++){
        Invert6x6_eigen(Y, Yinv, user);
        Invert6x6_eigen(Z, Zinv, user);

        for (i=0;i<dof;i++){
            for (j=0;j<dof;j++){
                Y[i][j]= 0.5*(Y[i][j] + Zinv[i][j]);
            }
        }

        for (i=0;i<dof;i++){
            for (j=0;j<dof;j++){
                Z[i][j]= 0.5*(Z[i][j] + Yinv[i][j]);
            }
        }
      }

    for (i=0;i<dof;i++){
        for (j=0;j<dof;j++){
            TauTempInv[i][j]= Z[i][j]/sqrt(taut);
        }
    }

    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeAMatrixConservation6x6"
PetscErrorCode ComputeAMatrixConservation6x6(PetscScalar u[],
                                          PetscReal (*A0inv)[6],PetscReal (*A1_c)[6],PetscReal (*A1_p)[6], PetscReal (*A1_pt)[6],
                                          PetscReal (*A2_c)[6],PetscReal (*A2_p)[6], PetscReal (*A2_pt)[6], PetscReal (*A3_c)[6],
                                          PetscReal (*A3_p)[6], PetscReal (*A3_pt)[6], PetscReal (*K)[3][4][4],
                                          PetscReal (*A1_cons)[6],PetscReal (*A2_cons)[6],PetscReal (*A3_cons)[6],PetscReal (*K_cons)[3][5][5],void *ctx)
{
    AppCtx *user = (AppCtx *)ctx;
    PetscInt  dof    = user->iga->dof;
    PetscInt i,j,l;

    for (i=0;i<dof;i++){
        for (j=0;j<dof;j++){
            for (l=0;l<dof;l++){
                A1_cons[i][j] += (A1_p[i][l]+A1_c[i][l]+A1_pt[i][l])*A0inv[l][j];
                A2_cons[i][j] += (A2_p[i][l]+A2_c[i][l]+A2_pt[i][l])*A0inv[l][j];
                A3_cons[i][j] += (A3_p[i][l]+A3_c[i][l]+A3_pt[i][l])*A0inv[l][j];
            }
        }
    }

    // We do not need to increase the dimensionality of K_{ijkl}, so leave this alone by only operating on the relevant DOF
    dof -= user->numLS_Fields;
    for (i=0;i<dof-1;i++){
        for (j=0;j<dof-1;j++){
            for (l=0;l<dof-1;l++){
                K_cons[0][0][i+1][j] += K[0][0][i][l]*A0inv[l+1][j];
                K_cons[0][1][i+1][j] += K[0][1][i][l]*A0inv[l+1][j];
                K_cons[0][2][i+1][j] += K[0][2][i][l]*A0inv[l+1][j];
                K_cons[1][0][i+1][j] += K[1][0][i][l]*A0inv[l+1][j];
                K_cons[1][1][i+1][j] += K[1][1][i][l]*A0inv[l+1][j];
                K_cons[1][2][i+1][j] += K[1][2][i][l]*A0inv[l+1][j];
                K_cons[2][0][i+1][j] += K[2][0][i][l]*A0inv[l+1][j];
                K_cons[2][1][i+1][j] += K[2][1][i][l]*A0inv[l+1][j];
                K_cons[2][2][i+1][j] += K[2][2][i][l]*A0inv[l+1][j];
            }
        }
    }

    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "DirectTau6x6"
PetscErrorCode DirectTau6x6(PetscReal G[3][3],
                         PetscReal dt,PetscScalar u[],
                         PetscScalar (*tau)[6],void *ctx,PetscReal (*A0inv)[6],PetscReal (*A1_cons)[6],PetscReal (*A2_cons)[6],PetscReal (*A3_cons)[6],PetscReal (*K_cons)[3][5][5],PetscReal *umi)
{
    // Compute Tau_SUPG for system with 6 DOF

    AppCtx *user     = (AppCtx *)ctx;
    PetscInt  dof    = user->iga->dof;


    //Tau for Conservation variables
    PetscReal TauTemp[6][6]   = {{0.0}};
    PetscReal TauTempInv[6][6]= {{0.0}};
    PetscReal taut            = 4.0/(dt*dt);
    PetscInt  aa,bb,ll;

    // Term 1
    for (aa=0;aa<dof;aa++)
        TauTemp[aa][aa] = taut;

    // Term 2
    for (aa=0;aa<dof;aa++)
        for (bb=0;bb<dof;bb++)
            for (ll=0;ll<dof;ll++){
                TauTemp[aa][bb] += G[0][0]*A1_cons[aa][ll]*A1_cons[ll][bb]
                +  G[0][1]*A1_cons[aa][ll]*A2_cons[ll][bb]
                +  G[0][2]*A1_cons[aa][ll]*A3_cons[ll][bb]
                +  G[1][0]*A2_cons[aa][ll]*A1_cons[ll][bb]
                +  G[1][1]*A2_cons[aa][ll]*A2_cons[ll][bb]
                +  G[1][2]*A2_cons[aa][ll]*A3_cons[ll][bb]
                +  G[2][0]*A3_cons[aa][ll]*A1_cons[ll][bb]
                +  G[2][1]*A3_cons[aa][ll]*A2_cons[ll][bb]
                +  G[2][2]*A3_cons[aa][ll]*A3_cons[ll][bb];
            }


    // Term 3
    // We do not need to increase the dimensionality of K_{ijkl}, so leave this alone by only operating on the relevant DOF
    dof -= user->numLS_Fields;
    PetscReal m_k=3.0;
    for (aa=0;aa<dof;aa++)
        for (bb=0;bb<dof;bb++)
  	  	  	   for (ll=0;ll<dof;ll++){

                   TauTemp[aa][bb] += m_k*G[0][0]*K_cons[0][0][aa][ll]*K_cons[0][0][ll][bb]*G[0][0]
                   +  m_k*G[0][0]*K_cons[0][1][aa][ll]*K_cons[0][0][ll][bb]*G[0][1]
                   +  m_k*G[0][0]*K_cons[0][2][aa][ll]*K_cons[0][0][ll][bb]*G[0][2]

                   +  m_k*G[0][1]*K_cons[1][0][aa][ll]*K_cons[0][0][ll][bb]*G[0][0]
                   +  m_k*G[0][1]*K_cons[1][1][aa][ll]*K_cons[0][0][ll][bb]*G[0][1]
                   +  m_k*G[0][1]*K_cons[1][2][aa][ll]*K_cons[0][0][ll][bb]*G[0][2]

                   +  m_k*G[0][2]*K_cons[2][0][aa][ll]*K_cons[0][0][ll][bb]*G[0][0]
                   +  m_k*G[0][2]*K_cons[2][1][aa][ll]*K_cons[0][0][ll][bb]*G[0][1]
                   +  m_k*G[0][2]*K_cons[2][2][aa][ll]*K_cons[0][0][ll][bb]*G[0][2]



                   +  m_k*G[0][0]*K_cons[0][0][aa][ll]*K_cons[0][1][ll][bb]*G[1][0]
                   +  m_k*G[0][0]*K_cons[0][1][aa][ll]*K_cons[0][1][ll][bb]*G[1][1]
                   +  m_k*G[0][0]*K_cons[0][2][aa][ll]*K_cons[0][1][ll][bb]*G[1][2]

                   +  m_k*G[0][1]*K_cons[1][0][aa][ll]*K_cons[0][1][ll][bb]*G[1][0]
                   +  m_k*G[0][1]*K_cons[1][1][aa][ll]*K_cons[0][1][ll][bb]*G[1][1]
                   +  m_k*G[0][1]*K_cons[1][2][aa][ll]*K_cons[0][1][ll][bb]*G[1][2]

                   +  m_k*G[0][2]*K_cons[2][0][aa][ll]*K_cons[0][1][ll][bb]*G[1][0]
                   +  m_k*G[0][2]*K_cons[2][1][aa][ll]*K_cons[0][1][ll][bb]*G[1][1]
                   +  m_k*G[0][2]*K_cons[2][2][aa][ll]*K_cons[0][1][ll][bb]*G[1][2]


                   +  m_k*G[0][0]*K_cons[0][0][aa][ll]*K_cons[0][2][ll][bb]*G[2][0]
                   +  m_k*G[0][0]*K_cons[0][1][aa][ll]*K_cons[0][2][ll][bb]*G[2][1]
                   +  m_k*G[0][0]*K_cons[0][2][aa][ll]*K_cons[0][2][ll][bb]*G[2][2]

                   +  m_k*G[0][1]*K_cons[1][0][aa][ll]*K_cons[0][2][ll][bb]*G[2][0]
                   +  m_k*G[0][1]*K_cons[1][1][aa][ll]*K_cons[0][2][ll][bb]*G[2][1]
                   +  m_k*G[0][1]*K_cons[1][2][aa][ll]*K_cons[0][2][ll][bb]*G[2][2]

                   +  m_k*G[0][2]*K_cons[2][0][aa][ll]*K_cons[0][2][ll][bb]*G[2][0]
                   +  m_k*G[0][2]*K_cons[2][1][aa][ll]*K_cons[0][2][ll][bb]*G[2][1]
                   +  m_k*G[0][2]*K_cons[2][2][aa][ll]*K_cons[0][2][ll][bb]*G[2][2]


                   +  m_k*G[1][0]*K_cons[0][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[1][0]*K_cons[0][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[1][0]*K_cons[0][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[1][1]*K_cons[1][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[1][1]*K_cons[1][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[1][1]*K_cons[1][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[1][2]*K_cons[2][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[1][2]*K_cons[2][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[1][2]*K_cons[2][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]


                   +  m_k*G[1][0]*K_cons[0][0][aa][ll]*K_cons[1][1][ll][bb]*G[1][0]
                   +  m_k*G[1][0]*K_cons[0][1][aa][ll]*K_cons[1][1][ll][bb]*G[1][1]
                   +  m_k*G[1][0]*K_cons[0][2][aa][ll]*K_cons[1][1][ll][bb]*G[1][2]

                   +  m_k*G[1][1]*K_cons[1][0][aa][ll]*K_cons[1][1][ll][bb]*G[1][0]
                   +  m_k*G[1][1]*K_cons[1][1][aa][ll]*K_cons[1][1][ll][bb]*G[1][1]
                   +  m_k*G[1][1]*K_cons[1][2][aa][ll]*K_cons[1][1][ll][bb]*G[1][2]

                   +  m_k*G[1][2]*K_cons[2][0][aa][ll]*K_cons[1][1][ll][bb]*G[1][0]
                   +  m_k*G[1][2]*K_cons[2][1][aa][ll]*K_cons[1][1][ll][bb]*G[1][1]
                   +  m_k*G[1][2]*K_cons[2][2][aa][ll]*K_cons[1][1][ll][bb]*G[1][2]


                   +  m_k*G[1][0]*K_cons[0][0][aa][ll]*K_cons[1][2][ll][bb]*G[2][0]
                   +  m_k*G[1][0]*K_cons[0][1][aa][ll]*K_cons[1][2][ll][bb]*G[2][1]
                   +  m_k*G[1][0]*K_cons[0][2][aa][ll]*K_cons[1][2][ll][bb]*G[2][2]

                   +  m_k*G[1][1]*K_cons[1][0][aa][ll]*K_cons[1][2][ll][bb]*G[2][0]
                   +  m_k*G[1][1]*K_cons[1][1][aa][ll]*K_cons[1][2][ll][bb]*G[2][1]
                   +  m_k*G[1][1]*K_cons[1][2][aa][ll]*K_cons[1][2][ll][bb]*G[2][2]

                   +  m_k*G[1][2]*K_cons[2][0][aa][ll]*K_cons[1][2][ll][bb]*G[2][0]
                   +  m_k*G[1][2]*K_cons[2][1][aa][ll]*K_cons[1][2][ll][bb]*G[2][1]
                   +  m_k*G[1][2]*K_cons[2][2][aa][ll]*K_cons[1][2][ll][bb]*G[2][2]



                   +  m_k*G[2][0]*K_cons[0][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[2][0]*K_cons[0][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[2][0]*K_cons[0][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]


                   +  m_k*G[2][1]*K_cons[1][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[2][1]*K_cons[1][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[2][1]*K_cons[1][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[2][2]*K_cons[2][0][aa][ll]*K_cons[1][0][ll][bb]*G[0][0]
                   +  m_k*G[2][2]*K_cons[2][1][aa][ll]*K_cons[1][0][ll][bb]*G[0][1]
                   +  m_k*G[2][2]*K_cons[2][2][aa][ll]*K_cons[1][0][ll][bb]*G[0][2]

                   +  m_k*G[2][0]*K_cons[0][0][aa][ll]*K_cons[2][1][ll][bb]*G[1][0]
                   +  m_k*G[2][0]*K_cons[0][1][aa][ll]*K_cons[2][1][ll][bb]*G[1][1]
                   +  m_k*G[2][0]*K_cons[0][2][aa][ll]*K_cons[2][1][ll][bb]*G[1][2]

                   +  m_k*G[2][1]*K_cons[1][0][aa][ll]*K_cons[2][1][ll][bb]*G[1][0]
                   +  m_k*G[2][1]*K_cons[1][1][aa][ll]*K_cons[2][1][ll][bb]*G[1][1]
                   +  m_k*G[2][1]*K_cons[1][2][aa][ll]*K_cons[2][1][ll][bb]*G[1][2]

                   +  m_k*G[2][2]*K_cons[2][0][aa][ll]*K_cons[2][1][ll][bb]*G[1][0]
                   +  m_k*G[2][2]*K_cons[2][1][aa][ll]*K_cons[2][1][ll][bb]*G[1][1]
                   +  m_k*G[2][2]*K_cons[2][2][aa][ll]*K_cons[2][1][ll][bb]*G[1][2]

                   +  m_k*G[2][0]*K_cons[0][0][aa][ll]*K_cons[2][2][ll][bb]*G[2][0]
                   +  m_k*G[2][0]*K_cons[0][1][aa][ll]*K_cons[2][2][ll][bb]*G[2][1]
                   +  m_k*G[2][0]*K_cons[0][2][aa][ll]*K_cons[2][2][ll][bb]*G[2][2]

                   +  m_k*G[2][1]*K_cons[1][0][aa][ll]*K_cons[2][2][ll][bb]*G[2][0]
                   +  m_k*G[2][1]*K_cons[1][1][aa][ll]*K_cons[2][2][ll][bb]*G[2][1]
                   +  m_k*G[2][1]*K_cons[1][2][aa][ll]*K_cons[2][2][ll][bb]*G[2][2]

                   +  m_k*G[2][2]*K_cons[2][0][aa][ll]*K_cons[2][2][ll][bb]*G[2][0]
                   +  m_k*G[2][2]*K_cons[2][1][aa][ll]*K_cons[2][2][ll][bb]*G[2][1]
                   +  m_k*G[2][2]*K_cons[2][2][aa][ll]*K_cons[2][2][ll][bb]*G[2][2];
               }

    // ()^(-0.5)
    Compute_square_root_inverse_6x6(TauTemp, TauTempInv, taut, user);

    //Transform to primitive variables
    // Reset DOF = 6 to transform back to the primitive variables
    dof += user->numLS_Fields;
    for (aa=0;aa<dof;aa++)
        for (bb=0;bb<dof;bb++){
            tau[aa][bb] = A0inv[aa][0]*TauTempInv[0][bb]
            +A0inv[aa][1]*TauTempInv[1][bb]
            +A0inv[aa][2]*TauTempInv[2][bb]
            +A0inv[aa][3]*TauTempInv[3][bb]
            +A0inv[aa][4]*TauTempInv[4][bb]
            +A0inv[aa][5]*TauTempInv[5][bb];
        }
    //

    return 0;
}

//////////////////////////////////////////////////////////////////


#undef __FUNCT__
#define __FUNCT__ "probePressureSignals76"
PetscErrorCode probePressureSignals76(AppCtx *user, PARAMETERS *par)
{
  PetscFunctionBegin;
  //PetscMPIInt rank;
  //MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscErrorCode ierr;
  FILE *fileResult;
  char filenameResults[50];

PetscReal pt[3] = {0.0};
PetscScalar uf[6] = {0.0};
IGAProbe prb;
ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);

pt[0] = 0.076/(sqrt(3)*user->Lx);
pt[1] = 0.076/(sqrt(3)*user->Ly);
pt[2] = 0.076/(sqrt(3)*user->Lz);

ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

PetscReal dens0  = 1000.0;
PetscReal P0     = 100000.0;
PetscReal B      = 3.31e8;
PetscReal N      = 7.15;
PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
PetscReal Pcr    = 22.02726;

PetscReal dens = uf[0];

PetscReal P        = Pcr;
PetscReal fprime   = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
PetscReal cs       = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
if(dens>rhoCR){
   P        = P0+B*(pow(dens/dens0,N))-B;
   fprime   = (1.0/dens)*B*N*(pow(dens/dens0,N));
   cs       = (1.0/dens)*B*N*(pow(dens/dens0,N));
 }

IGAProbeDestroy(&prb);
//PetscPrintf(PETSC_COMM_WORLD, "Probe density = %e Pressure at Probe = %e\n", dens, P);
//PetscPrintf(PETSC_COMM_WORLD, "u_z at Probe = %e\n", uf[3]);

sprintf(filenameResults,"./Output_ProbeData/pressureSignal76mm.csv");

if(par->stepNumber>0)
  fileResult=fopen(filenameResults,"a");
else
  fileResult=fopen(filenameResults,"wt");
if (fileResult == NULL){
  printf("Error opening file (Probe Output)!\n");
  exit(1);
}

PetscFPrintf(PETSC_COMM_WORLD,fileResult,"%.5e, %.5e ,%.5e \n", par->currentTime, P, dens);

fclose(fileResult);

PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "probePressureSignals114"
PetscErrorCode probePressureSignals114(AppCtx *user, PARAMETERS *par)
{
  PetscFunctionBegin;
  //PetscMPIInt rank;
  //MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscErrorCode ierr;
  FILE *fileResult;
  char filenameResults[50];

PetscReal pt[3] = {0.0};
PetscScalar uf[6] = {0.0};
IGAProbe prb;
ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);

pt[0] = 0.114/(sqrt(3)*user->Lx);
pt[1] = 0.114/(sqrt(3)*user->Ly);
pt[2] = 0.114/(sqrt(3)*user->Lz);

ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

PetscReal dens0  = 1000.0;
PetscReal P0     = 100000.0;
PetscReal B      = 3.31e8;
PetscReal N      = 7.15;
PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
PetscReal Pcr    = 22.02726;

PetscReal dens = uf[0];

PetscReal P        = Pcr;
PetscReal fprime   = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
PetscReal cs       = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
if(dens>rhoCR){
   P        = P0+B*(pow(dens/dens0,N))-B;
   fprime   = (1.0/dens)*B*N*(pow(dens/dens0,N));
   cs       = (1.0/dens)*B*N*(pow(dens/dens0,N));
 }

IGAProbeDestroy(&prb);
//PetscPrintf(PETSC_COMM_WORLD, "Probe density = %e Pressure at Probe = %e\n", dens, P);
//PetscPrintf(PETSC_COMM_WORLD, "u_z at Probe = %e\n", uf[3]);

sprintf(filenameResults,"./Output_ProbeData/pressureSignal114mm.csv");

if(par->stepNumber>0)
  fileResult=fopen(filenameResults,"a");
else
  fileResult=fopen(filenameResults,"wt");
if (fileResult == NULL){
  printf("Error opening file (Probe Output)!\n");
  exit(1);
}

PetscFPrintf(PETSC_COMM_WORLD,fileResult,"%.5e, %.5e ,%.5e \n", par->currentTime, P, dens);

fclose(fileResult);

PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "probePressureSignals152"
PetscErrorCode probePressureSignals152(AppCtx *user, PARAMETERS *par)
{
  PetscFunctionBegin;
  //PetscMPIInt rank;
  //MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscErrorCode ierr;
  FILE *fileResult;
  char filenameResults[50];

PetscReal pt[3] = {0.0};
PetscScalar uf[6] = {0.0};
IGAProbe prb;
ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);

pt[0] = 0.152/(sqrt(3)*user->Lx);
pt[1] = 0.152/(sqrt(3)*user->Ly);
pt[2] = 0.152/(sqrt(3)*user->Lz);

ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

PetscReal dens0  = 1000.0;
PetscReal P0     = 100000.0;
PetscReal B      = 3.31e8;
PetscReal N      = 7.15;
PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
PetscReal Pcr    = 22.02726;

PetscReal dens = uf[0];

PetscReal P        = Pcr;
PetscReal fprime   = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
PetscReal cs       = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
if(dens>rhoCR){
   P        = P0+B*(pow(dens/dens0,N))-B;
   fprime   = (1.0/dens)*B*N*(pow(dens/dens0,N));
   cs       = (1.0/dens)*B*N*(pow(dens/dens0,N));
 }

IGAProbeDestroy(&prb);
//PetscPrintf(PETSC_COMM_WORLD, "Probe density = %e Pressure at Probe = %e\n", dens, P);
//PetscPrintf(PETSC_COMM_WORLD, "u_z at Probe = %e\n", uf[3]);

sprintf(filenameResults,"./Output_ProbeData/pressureSignal152mm.csv");

if(par->stepNumber>0)
  fileResult=fopen(filenameResults,"a");
else
  fileResult=fopen(filenameResults,"wt");
if (fileResult == NULL){
  printf("Error opening file (Probe Output)!\n");
  exit(1);
}

PetscFPrintf(PETSC_COMM_WORLD,fileResult,"%.5e, %.5e ,%.5e \n", par->currentTime, P, dens);

fclose(fileResult);

PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "outputTimeData"
PetscErrorCode outputTimeData(AppCtx *user, PARAMETERS *par)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  FILE *fileResult;
  char filenameResults[50];
  sprintf(filenameResults,"./timeData.csv");

if(par->stepNumber>0)
  fileResult=fopen(filenameResults,"a");
else
  fileResult=fopen(filenameResults,"wt");
if (fileResult == NULL){
  printf("Error opening file (Probe Output)!\n");
  exit(1);
}

PetscFPrintf(PETSC_COMM_WORLD,fileResult,"%.5e, %5e \n", par->currentTime, par->timeStep);
fclose(fileResult);

PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeCurrentExplosiveVolume"
PetscErrorCode ComputeCurrentExplosiveVolume(AppCtx *user, PARAMETERS *par, ParticleManager &manager)
{
  PetscFunctionBegin;

  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscErrorCode ierr;

  PetscScalar uf[5] = {0.0};
  PetscReal pt[3] = {0.0};
  PetscInt i;
  user->totalCurrentExplosiveVolume = 0.0;
  user->totalExplosiveMass = 0.0;

  // Get the pressures at the nodes to output at the same time:
  PetscReal dens0  = 1770.0;
  PetscReal P0     = 100000.0;
  PetscReal A      = 7.78e11;
  PetscReal B      = 7.07e9;
  PetscReal C      = 1.62e9;
  PetscReal R1     = 4.485;
  PetscReal R2     = 1.068;
  PetscReal omega  = 0.3;
  PetscReal E0     = 5.93e6;
  PetscReal Pcr    = 2.0e11;
  PetscReal nu     = 0.0;
  PetscReal Ptest  = A*exp(-R1*nu) + B*exp(-R2*nu) + C/(pow(nu, 1+omega));

  PetscReal P        = Pcr;
  PetscReal fprime   = 0.0;
  PetscReal cs       = 0.0;
  IGAProbe  prb;

  ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);
  IGAProbeSetCollective(prb, PETSC_FALSE);

  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){

    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);

    pt[0] =  info.currentCoord[0]/user->Lx;
    pt[1] =  info.currentCoord[1]/user->Ly;
    pt[2] =  info.currentCoord[2]/user->Lz;

    if((!info.isTask)  && (fd.Inside == 1)){
      if (!fd.material == 0){

    ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
    ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

    PetscScalar dens = uf[0];
    PetscScalar T    = uf[4];

    if(user->stepNumber == 0 && user->it == 0){
      fd.nodalDensityInitial = dens;
      if(fd.material==1){
      user->totalExplosiveMass += dens*fd.nodalVolume;
      }
    }

    fd.nodalDensity = dens;
    if(dens<=0.0){
      PetscPrintf(PETSC_COMM_SELF,"Density < 0, Current Explosive Density Update error \n pt = %e %e %e on rank %d with density = %e material = %d\n", pt[0], pt[1], pt[2], rank, dens, fd.material);
      exit(0);
    }

    if(fd.material==1){
      fd.nodalVolume = fd.nodalVolumeInitial * fd.nodalDensityInitial/(dens+10E-15);
    }

    if((user->stepNumber > 0 || user->it > 0) && fd.material==1){
      user->totalExplosiveMass+=fd.nodalVolume*dens;
    }

    if(fd.nodalVolume<=(10E-15)){
      PetscPrintf(PETSC_COMM_SELF,"Volume <= 0, Current Explosive Volume Update error \n rank = %d \n", rank);
      exit(0);
    }

    if(fd.material==1){
    user->totalCurrentExplosiveVolume += fd.nodalVolume;
    nu     =  dens0/fd.nodalDensity;
    Ptest  = A*exp(-R1*nu) + B*exp(-R2*nu) + C/(pow(nu, 1+omega));
    P      = Pcr;
    if(PETSC_FALSE){
      P        = Ptest;
    }else{
      P        = A*(1-omega/(R1*nu))*exp(-R1*nu) + B*(1-omega/(R2*nu))*exp(-R2*nu) + omega*dens0*E0/nu;
    }
    fd.nodalPressure = P;
    }


    put(fd_property,v,fd);
    put(info_property,v,info);

          }
        }
    }

  IGAProbeDestroy(&prb);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "updateVolumeAndDensity"
PetscErrorCode updateVolumeAndDensity(PARAMETERS *par,
                                      AppCtx *user,
                                      ParticleManager &manager)
{
  // Perform Volume and density update for the foreground particles associated with PD
  // based on determinant of alpha level deformation gradient <- alpha level velocity gradient

PetscFunctionBegin;
pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
for(auto it=its.first; it != its.second; ++it){
  Edge edge = *it;
  Vertex v = target(edge,manager.graph);
  ParticleInfo info = get(info_property,v);
  FieldData fd = get(fd_property,v);
  if(fd.material==0 && !info.isTask){

  fd.alphaNodalVolume = fd.nodalVolume;// * fd.determinantAlphaDeformationGradient;
  //fd.nodalVolume = fd.nodalVolume;// * fd.determinantCurrentDeformationGradient;

  fd.alphaNodalDensity = par->density;// / fd.determinantAlphaDeformationGradient;
  fd.nodalDensity = par->density;// / fd.determinantCurrentDeformationGradient;

  }
  put(fd_property,v,fd);
}
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "setReferenceAsCurrent"
PetscErrorCode setReferenceAsCurrent(PARAMETERS *par, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt j;

  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){
  Edge edge = *it;
  Vertex v = target(edge,manager.graph);
  ParticleInfo info = get(info_property,v);
  FieldData fd = get(fd_property,v);
  if(!info.isTask){
  for(j=0;j<3;j++){
      fd.referenceCoord[j]=info.currentCoord[j];
  }
}
  put(fd_property,v,fd);
  }
PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "setReferenceAsCurrentVolume"
PetscErrorCode setReferenceAsCurrentVolume(PARAMETERS *par, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt i;

  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);
    if(!info.isTask){
    if(fd.material==0){
      fd.referenceNodalVolume = fd.nodalVolume;
      fd.referenceDensity = fd.nodalDensity;
    }else{
      // These quantities are only used by PD nodes, so we can set them to zero for others
      // Really, should separate the data structures
      fd.referenceNodalVolume = 0.0;
      fd.referenceDensity = 0.0;
    }
  }
  put(fd_property, v, fd);
  }

  PetscFunctionReturn(0);
}
//// End Stabilization ////

//// Residual (Fluid & Solid) Integrand Definitions ////
// Material 1 : RDX
// Material 0 : Immersed PD particle
#undef __FUNCT__
#define __FUNCT__ "computeInertia"
PetscErrorCode computeInertia(PARAMETERS *par,
                              AppCtx *user,
                              ParticleManager &manager)
{
// For particles associated with immersed PD structure, evaluate
// Total physical acceleration through the lagrangian update using displ. and
// vel. after the peridigm->updateState() computation.
PetscFunctionBegin;
pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
for(auto it=its.first; it != its.second; ++it){
  Edge edge = *it;
  Vertex v = target(edge,manager.graph);
  ParticleInfo info = get(info_property,v);
  FieldData fd = get(fd_property,v);
  if(fd.material==0 && !info.isTask){
    fd.inertia[0] = par->density * fd.totalPhysicalAcceleration[0]; //Assumes force and inertia density (Kg/(m^2s^2))
    fd.inertia[1] = par->density * fd.totalPhysicalAcceleration[1];
    fd.inertia[2] = par->density * fd.totalPhysicalAcceleration[2];
  }
  put(fd_property,v,fd);
}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "computeSolidResidualStrong"
PetscErrorCode computeSolidResidualStrong(PARAMETERS *par,
                                          AppCtx *user, ParticleManager &manager)
{
  PetscFunctionBegin;

  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){

  Edge edge = *it;
  Vertex v = target(edge,manager.graph);
  ParticleInfo info = get(info_property,v);
  FieldData fd = get(fd_property,v);

  if(fd.material==0 && !info.isTask){
    for (int j=0 ; j<user->iga->dim ; j++){
      fd.residual[j] = fd.inertia[j] - fd.internalForce[j] - fd.bodyForce[j];  //Assumes force and inertia density (Kg/(m^2s^2))
    }
  }
  put(fd_property,v,fd);
  put(info_property,v,info);
  }

  PetscFunctionReturn(0);
}



#undef  __FUNCT__
#define __FUNCT__ "Residual"
PetscErrorCode Residual(IGAPoint pnt,
                        PetscReal dt,
                        const PetscScalar *V,
                        PetscReal t,
                        const PetscScalar *U,
                        PetscScalar *Re,
                        void *ctx)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx *)ctx;

  if (pnt->atboundary){ // IGA point on the boundary

    PetscInt dof = pnt->dof;
    PetscInt dim = pnt->dim;
    PetscScalar u[dof];

    IGAPointFormValue(pnt,U,&u[0]);
    PetscScalar dens     = u[0];
    PetscScalar alpha_1  = u[5];
    PetscScalar alpha_2  = 1.0 - alpha_1;

    PetscReal P_1, fprime_1, cs_1, energy_1;
    PetscReal P_2, fprime_2, cs_2, energy_2;
    PetscReal P;
    PetscReal F1[6]={0.0};
    PetscReal F2[6]={0.0};
    PetscReal F3[6]={0.0};

    P_2 = TaitEq_WATER(dens, &fprime_2, &cs_2);
    P_1 = JWL_RDX(dens, &fprime_1, &cs_1);
    P   = alpha_1*P_1 + alpha_2*P_2;
    F1[1]          = P;
    F2[2]          = P;
    F3[3]          = P;

    PetscReal   *N0 = pnt->shape[0];
    PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;

    for (int a=0 ; a<pnt->nen ; a++) {
      for (int i=0;i<6;i++)
      R[a][i] += N0[a]*F1[i]*pnt->normal[0] + N0[a]*F2[i]*pnt->normal[1] + N0[a]*F3[i]*pnt->normal[2];
    }
  }

  else{ // IGA point in the interior
  PetscInt i,j,l,m;
  PetscInt  dof     = pnt->dof;
  PetscInt  dim     = pnt->dim;
  PetscReal lamda_2 = user->lamda;
  PetscReal mu_2    = user->mu;
  PetscReal kappa_2 = user->kappa;
  PetscReal Cv_2    = user->Cv;
  PetscReal chi_2   = lamda_2+2*mu_2;

  PetscScalar u[dof], u_t[dof];
  PetscReal      G[3][3] = {{0.0}};
  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormValue(pnt,V,&u_t[0]);

  PetscScalar dens    = u[0];
  PetscScalar ux      = u[1];
  PetscScalar uy      = u[2];
  PetscScalar uz      = u[3];
  PetscScalar temp    = u[4];
  PetscScalar alpha_1 = u[5];
  PetscScalar alpha_2 = 1.0 - alpha_1;

  PetscScalar grad_u[dof][dim];
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  PetscReal InvGradMap[dim][dim];
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);

  for (i=0;i<dim;i++){
    for (j=0;j<dim;j++){
      for (l=0;l<dim;l++){
       G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
     }
   }
  }
 PetscReal gradPhiGgradPhi = 0.0;
 for (int i=0;i<dim;i++){
   for (int j=0;j<dim;j++){
     gradPhiGgradPhi += grad_u[5][i]*G[i][j]*grad_u[5][j];
   }
 }
 PetscReal localWidth = 0.5*sqrt(grad_u[5][0]*grad_u[5][0]+grad_u[5][1]*grad_u[5][1]+grad_u[5][2]*grad_u[5][2])/sqrt(gradPhiGgradPhi);

 //// Fluid Property reconstruction
 PetscReal P_1, fprime_1, cs_1, energy_1;
 PetscReal P_2, fprime_2, cs_2, energy_2;

 P_2      = TaitEq_WATER(dens, &fprime_2, &cs_2);
 energy_2 = Energy_WATER(temp);
 P_1      = JWL_RDX(dens, &fprime_1, &cs_1);
 energy_1 = Energy_RDX(temp);

 PetscReal P, fprime, cs, e;
 P      = alpha_1*P_1      + alpha_2*P_2;
 fprime = alpha_1*fprime_1 + alpha_2*fprime_2;
 cs     = alpha_1*cs_1     + alpha_2*cs_2;
 e      = alpha_1*energy_1 + alpha_2*energy_2;
 ////////////////////////////////////

 // Reconstruct physical properties of the fluid (the current second fluid is inviscid and kappa = 0)
 PetscReal lamda = alpha_2*lamda_2;
 PetscReal mu    = alpha_2*mu_2;
 PetscReal kappa = alpha_2*kappa_2;
 PetscReal Cv    = alpha_2*Cv_2 + alpha_1*143.3;
 PetscReal chi   = alpha_2*chi_2;
 //////////////////////////////////////////////////////////

 PetscReal      umi[3]     = {0.0};
 PetscScalar    tau[6][6]  = {{0.0}};
 PetscReal      A1_c[6][6] = {{0.0}};
 PetscReal      A1_p[6][6] = {{0.0}};
 PetscReal      A2_c[6][6] = {{0.0}};
 PetscReal      A2_p[6][6] = {{0.0}};
 PetscReal      A3_c[6][6] = {{0.0}};
 PetscReal      A3_p[6][6] = {{0.0}};
 PetscReal      A3_pt[6][6]= {{0.0}};
 PetscReal      A2_pt[6][6]= {{0.0}};
 PetscReal      A1_pt[6][6]= {{0.0}};
 PetscReal      A0[6][6]   = {{0.0}};
 PetscReal      A0inv[6][6]= {{0.0}};
 PetscReal      K[3][3][4][4] = {{{{0.0}}}};

 //Tau_{ij} = lambda*(div(u))+mu*(symmetric grad u)
 PetscReal t11, t12, t13, t21, t22, t23, t31, t32, t33;
 t11 = mu*(grad_u[1][0]+grad_u[1][0])+lamda*(grad_u[1][0]+grad_u[2][1]+grad_u[3][2]);
 t22 = mu*(grad_u[2][1]+grad_u[2][1])+lamda*(grad_u[1][0]+grad_u[2][1]+grad_u[3][2]);
 t33 = mu*(grad_u[3][2]+grad_u[3][2])+lamda*(grad_u[1][0]+grad_u[2][1]+grad_u[3][2]);
 t12 = mu*(grad_u[1][1]+grad_u[2][0]);
 t13 = mu*(grad_u[1][2]+grad_u[3][0]);
 t21 = mu*(grad_u[2][0]+grad_u[1][1]);
 t23 = mu*(grad_u[2][2]+grad_u[3][1]);
 t31 = mu*(grad_u[3][0]+grad_u[1][0]);
 t32 = mu*(grad_u[3][1]+grad_u[2][2]);
 //

 // Density Primitive A0: Convective LS formualtion
 A0[0][0] = 1.0;
 A0[1][0] = ux;
 A0[1][1] = dens;
 A0[2][0] = uy;
 A0[2][2] = dens;
 A0[3][0] = uz;
 A0[3][3] = dens;
 A0[4][0] = e;
 A0[4][4] = dens*e/temp;
 A0[4][5] = dens*(energy_1-energy_2);
 A0[5][0] = alpha_1;
 A0[5][5] = dens;

 // Inverse of A0:
 A0inv[0][0] =  1.0;
 A0inv[1][0] = -ux/dens;
 A0inv[1][1] =  1.0/dens;
 A0inv[2][0] = -uy/dens;
 A0inv[2][2] =  1.0/dens;
 A0inv[3][0] = -uz/dens;
 A0inv[3][3] =  1.0/dens;
 A0inv[4][0] =  (-e*temp+(energy_1-energy_2)*temp*(alpha_1-alpha_2))/(dens*e);
 A0inv[4][4] =  temp/(dens*e);
 A0inv[4][5] = (-energy_1+energy_2)*temp/(dens*e);
 A0inv[5][0] = -alpha_1/dens;
 A0inv[5][5] =  1.0/dens;

 // Compute A1_p/adv
 A1_c[0][0] = ux;
 A1_c[0][1] = dens;
 A1_c[1][0] = ux*ux;
 A1_c[1][1] = 2.0*dens*ux;
 A1_c[2][0] = ux*uy;
 A1_c[2][1] = dens*uy;
 A1_c[2][2] = dens*ux;
 A1_c[3][0] = ux*uz;
 A1_c[3][1] = dens*uz;
 A1_c[3][3] = dens*ux;
 A1_c[4][0] = e*ux;
 A1_c[4][1] = dens*e;
 A1_c[4][4] = dens*ux*e/temp;
 A1_c[4][5] = ux*dens*(energy_1-energy_2);
 A1_c[5][0] = ux*alpha_1;
 A1_c[5][1] = alpha_1*dens;
 A1_c[5][5] = ux*dens;

 // A_1^p
 A1_p[1][0] = fprime+alpha_2*(P_2-P_1)/dens;
 A1_p[1][5] =  (P_1-P_2);

 //A_1^{SP}
 A1_pt[4][1] = P-t11;
 A1_pt[4][2] = -t12;
 A1_pt[4][3] = -t13;

 // Compute A2_p/adv
 A2_c[0][0] = uy;
 A2_c[0][2] = dens;
 A2_c[1][0] = ux*uy;
 A2_c[1][1] = dens*uy;
 A2_c[1][2] = dens*ux;
 A2_c[2][0] = uy*uy;
 A2_c[2][2] = 2.0*dens*uy;
 A2_c[3][0] = uy*uz;
 A2_c[3][2] = dens*uz;
 A2_c[3][3] = dens*uy;
 A2_c[4][0] = uy*e;
 A2_c[4][2] = dens*e;
 A2_c[4][4] = dens*uy*e/temp;
 A2_c[4][5] = uy*dens*(energy_1-energy_2);
 A2_c[5][0] = uy*alpha_1;
 A2_c[5][2] = dens*alpha_1;
 A2_c[5][5] = dens*uy;

 // A_2^p
 A2_p[2][0] = fprime+alpha_2*(P_2-P_1)/dens;
 A2_p[2][5] = (P_1-P_2);

 //A_2^{SP}
 A2_pt[4][1] = -t21;
 A2_pt[4][2] = P-t22;
 A2_pt[4][3] = -t23;

 // Compute A3_adv
 A3_c[0][0] = uz;
 A3_c[0][3] = dens;
 A3_c[1][0] = ux*uz;
 A3_c[1][1] = dens*uz;
 A3_c[1][3] = dens*ux;
 A3_c[2][0] = uy*uz;
 A3_c[2][2] = dens*uz;
 A3_c[2][3] = dens*uy;
 A3_c[3][0] = uz*uz;
 A3_c[3][3] = 2.0*dens*uz;
 A3_c[4][0] = e*uz;
 A3_c[4][3] = e*dens;
 A3_c[4][4] = dens*e*uz/temp;
 A3_c[4][5] = uz*dens*(energy_1-energy_2);
 A3_c[5][0] = uz*alpha_1;
 A3_c[5][3] = dens*alpha_1;
 A3_c[5][5] = uz*dens;

 //A_3^p
 A3_p[3][0] = fprime+alpha_2*(P_2-P_1)/dens;
 A3_p[3][5] = (P_1-P_2);

 //A_3^{SP}
 A3_pt[4][1] = -t31;
 A3_pt[4][2] = -t32;
 A3_pt[4][3] = P-t33;

 // Viscous terms, K_{ij{kl}}, reduced form, omitting
 // first row and first column = 0;
 // chi = lambda+2*mu

 //K_11kl
 K[0][0][0][0]=chi;
 K[0][0][1][1]=mu;
 K[0][0][2][2]=mu;
 K[0][0][3][3]=kappa;
 //K_22kl
 K[1][1][0][0]=mu;
 K[1][1][1][1]=chi;
 K[1][1][2][2]=mu;
 K[1][1][3][3]=kappa;
 //K_33kl
 K[2][2][0][0]=mu;
 K[2][2][1][1]=mu;
 K[2][2][2][2]=chi;
 K[2][2][3][3]=kappa;

 //K_12kl
 K[0][1][0][1]=lamda;
 K[0][1][1][0]=mu;
 //K_13kl
 K[0][2][0][2]=lamda;
 K[0][2][2][0]=mu;

 //K_21kl
 K[1][0][0][1]=mu;
 K[1][0][1][0]=lamda;
 //K_23kl
 K[1][2][1][2]=lamda;
 K[1][2][2][1]=mu;

 //K_31kl
 K[2][0][0][2]=mu;
 K[2][0][2][0]=lamda;
 //K_32kl
 K[2][1][1][2]=mu;
 K[2][1][2][1]=lamda;

PetscReal F1[6]={0.0};
PetscReal F2[6]={0.0};
PetscReal F3[6]={0.0};
F1[1] = P;
F2[2] = P;
F3[3] = P;

   //Direct Tau for 6 DOF
   PetscReal A1_cons[6][6] = {{0.0}};
   PetscReal A2_cons[6][6] = {{0.0}};
   PetscReal A3_cons[6][6] = {{0.0}};
   PetscReal K_cons[3][3][5][5] = {{{{0.0}}}};

   // Compute the SUPG parameter for 6x6 system
   ierr = ComputeAMatrixConservation6x6(u, A0inv, A1_c, A1_p, A1_pt,A2_c,A2_p,A2_pt,A3_c,A3_p,A3_pt,K,A1_cons,A2_cons,A3_cons,K_cons,user);CHKERRQ(ierr);
   ierr = DirectTau6x6(G,dt,u,tau,user,A0inv,A1_cons,A2_cons,A3_cons,K_cons,umi);CHKERRQ(ierr);

   PetscReal  *N0        = pnt->shape[0];
   PetscReal  (*N1)[dim] = (PetscReal (*)[dim]) pnt->shape[1];
   PetscReal  Res[6]     = {0.0};

 for (i=0;i<dof;i++){
 for (j=0;j<dof;j++){
       Res[i] += A0[i][j]*u_t[j] + (A1_c[i][j]+A1_p[i][j]+A1_pt[i][j])*grad_u[j][0]
       + (A2_c[i][j]+A2_p[i][j]+A2_pt[i][j])*grad_u[j][1] + (A3_c[i][j]+A3_p[i][j]+A3_pt[i][j])*grad_u[j][2];
 }
 }

 //Stabilization terms DC
  PetscReal hu[4] = {0.0};
  PetscReal A0gradY[6][3]={{0.0}};
  PetscReal tau_m,tau_c,tau_t,tau_ls;

  for (j=0;j<dim;j++)
  for (i=0;i<dof;i++)
  for (l=0;l<dof;l++)
  {
   A0gradY[i][j] += A0[i][l]*grad_u[l][j];
  }

  for (i=0;i<dim;i++){
  for (j=0;j<dim;j++){
   hu[0] += A0gradY[0][i]*G[i][j]*A0gradY[0][j];
   hu[1] += A0gradY[1][i]*G[i][j]*A0gradY[1][j];
   hu[1] += A0gradY[2][i]*G[i][j]*A0gradY[2][j];
   hu[1] += A0gradY[3][i]*G[i][j]*A0gradY[3][j];
   hu[2] += A0gradY[4][i]*G[i][j]*A0gradY[4][j];
   hu[3] += A0gradY[5][i]*G[i][j]*A0gradY[5][j];
  }
  }

  PetscReal Ginv[3][3]={{0.0}};
  PetscReal det = G[0][0] * (G[1][1] * G[2][2] - G[2][1] * G[1][2]) -
                  G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) +
                  G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);

  PetscReal invdet = 1.0 / det;

  Ginv[0][0] = (G[1][1] * G[2][2] - G[2][1] * G[1][2]) * invdet;
  Ginv[0][1] = (G[0][2] * G[2][1] - G[0][1] * G[2][2]) * invdet;
  Ginv[0][2] = (G[0][1] * G[1][2] - G[0][2] * G[1][1]) * invdet;
  Ginv[1][0] = (G[1][2] * G[2][0] - G[1][0] * G[2][2]) * invdet;
  Ginv[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) * invdet;
  Ginv[1][2] = (G[1][0] * G[0][2] - G[0][0] * G[1][2]) * invdet;
  Ginv[2][0] = (G[1][0] * G[2][1] - G[2][0] * G[1][1]) * invdet;
  Ginv[2][1] = (G[2][0] * G[0][1] - G[0][0] * G[2][1]) * invdet;
  Ginv[2][2] = (G[0][0] * G[1][1] - G[1][0] * G[0][1]) * invdet;

 PetscReal uGu = 0.0;
 for(i=0;i<dim;i++){
   for(j=0;j<dim;j++){
     uGu += u[i+1]*Ginv[i][j]*u[j+1];
   }
 }

 PetscReal k_cap = sqrt(uGu + cs*(Ginv[0][0]+Ginv[1][1]+Ginv[2][2]));
 PetscReal eps = 0.0;
 PetscReal DC = 2.0;

 // tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs + (Res[4]*Res[4])/cs)/
 // sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + hu[2]/cs+1e-15);
 /*tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
 sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);*/

 tau_c = sqrt((ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs + cs*Res[5]*Res[5])/
 sqrt((ux*ux+uy*uy+uz*uz)*hu[1]/cs + cs*hu[3] + 1e-15);

 tau_ls = sqrt(Res[5]*Res[5])/sqrt(hu[3]+1e-15);

 if(DC*tau_c > k_cap){tau_c = k_cap;}else{tau_c = DC*tau_c;}
 if(DC*tau_ls > k_cap){tau_ls = k_cap;}else{tau_ls = DC*tau_ls;}
 tau_m = tau_c;
 tau_t = tau_c;

 PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
 PetscInt a,nen=pnt->nen;

 for (a=0; a<nen; a++) {
   PetscReal Na    = N0[a];

   R[a][0]  = 0.0;
   R[a][1]  = 0.0;
   R[a][2]  = 0.0;
   R[a][3]  = 0.0;
   R[a][4]  = 0.0;
   R[a][5]  = 0.0;

   for (i=0;i<dof;i++){
   for (j=0;j<dof;j++){
     R[a][i] += (A0[i][j]*u_t[j]
     + (A1_c[i][j]+A1_pt[i][j])*grad_u[j][0]
     + (A2_c[i][j]+A2_pt[i][j])*grad_u[j][1]
     + (A3_c[i][j]+A3_pt[i][j])*grad_u[j][2])*Na;
   }
   }

   //Penalty term on alpha
   //if(alpha_1>1.0){R[a][5] += 2.0*dens*(alpha_1-1.0)*Na;}

   for (i=0;i<dof;i++){
         R[a][i] += -N1[a][0]*F1[i] - N1[a][1]*F2[i] - N1[a][2]*F3[i];
   }

   for (l=0;l<dim;l++){
   for (m=0;m<dim;m++){
     for (i=0;i<dof-2;i++){
       for (j=0;j<dof-2;j++){
               R[a][i+1] += N1[a][l]*K[l][m][i][j]*grad_u[j+1][m];   //ok
       }
     }
   }
   }

//Stabilization terms SUPG & DC
   for (i=0;i<dof;i++){
   for (j=0;j<dof;j++){
   for (l=0;l<dof;l++){
       R[a][i] += ((A1_c[i][j]+A1_p[i][j]+A1_pt[i][j])*N1[a][0]
       + (A2_c[i][j]+A2_p[i][j]+A2_pt[i][j])*N1[a][1]
       + (A3_c[i][j]+A3_p[i][j]+A3_pt[i][j])*N1[a][2])*tau[j][l]*Res[l];
   }
   }
   }

   for (i=0;i<dim;i++)
   {
               R[a][0] +=N1[a][i]*tau_c*A0gradY[0][i];
               R[a][1] +=N1[a][i]*tau_m*A0gradY[1][i];
               R[a][2] +=N1[a][i]*tau_m*A0gradY[2][i];
               R[a][3] +=N1[a][i]*tau_m*A0gradY[3][i];
               R[a][4] +=N1[a][i]*tau_t*A0gradY[4][i];
               R[a][5] +=N1[a][i]*tau_c*A0gradY[5][i];
   }
   }
  }
  return 0;
}

 #undef  __FUNCT__
 #define __FUNCT__ "Residual_massCalc"
 PetscErrorCode Residual_massCalc(IGAPoint pnt,
                         PetscReal dt,
                         const PetscScalar *V,
                         PetscReal t,
                         const PetscScalar *U,
                         PetscScalar *Re,
                         void *ctx)
 {
   PetscErrorCode ierr;
   AppCtx *user = (AppCtx *)ctx;

  if(!pnt->atboundary){ // IGA point in the interior
   PetscInt i,j,l,m;
   PetscInt  dof     = pnt->dof;
   PetscInt  dim     = pnt->dim;

   PetscScalar u[dof];
   IGAPointFormValue(pnt,U,&u[0]);
   PetscScalar dens    = u[0];
   PetscScalar alpha_1 = u[5];

  PetscScalar (*R)[dof]  = (PetscScalar (*)[dof])Re;
  PetscInt    nen        = pnt->nen;
  PetscReal   *N0        = pnt->shape[0];

  for (int a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    R[a][0]  = 0.0;
    if(alpha_1>=0.5) //Use the definition of the interface (H = 0.5 <-> phi = 0) to compute//The initial mass of the explosive.
    R[a][0] += alpha_1*Na*dens;
  }

  }
   return 0;
 }

 #undef  __FUNCT__
 #define __FUNCT__ "Residual_LS"
 PetscErrorCode Residual_LS(IGAPoint pnt,
                         PetscReal dt,
                         const PetscScalar *V,
                         PetscReal t,
                         const PetscScalar *U,
                         PetscScalar *Re,
                         const PetscScalar *U_main,
                         IGAPoint pnt_main,
                         PARAMETERS *par,
                         void *ctx)
 {
     PetscErrorCode ierr;
     AppCtx *user = (AppCtx *)ctx;
     PetscInt i,j,l;
     PetscInt  dof   = pnt->dof;
     PetscInt  dim   = pnt->dim;
     PetscScalar u[dof], u_t[dof], grad_u[dof][dim];
     PetscReal G[3][3] = {{0.0}};
     PetscScalar u_main[6];
     PetscReal InvGradMap[dim][dim];
     PetscReal convVel[3] = {0.0};

     IGAPointFormValue(pnt,U,&u[0]);
     IGAPointFormValue(pnt,V,&u_t[0]);
     IGAPointFormGrad (pnt,U,&grad_u[0][0]);
     IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);
     IGAPointFormValue(pnt_main,U_main,&u_main[0]);

     for (i=0;i<dim;i++){
       for (j=0;j<dim;j++){
         for (l=0;l<dim;l++){
           G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
         }
       }
     }
     PetscReal Ginv[3][3]={{0.0}};
     PetscReal det = G[0][0] * (G[1][1] * G[2][2] - G[2][1] * G[1][2]) -
                     G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) +
                     G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);

     PetscReal invdet = 1.0 / det;

     Ginv[0][0] = (G[1][1] * G[2][2] - G[2][1] * G[1][2]) * invdet;
     Ginv[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) * invdet;
     Ginv[2][2] = (G[0][0] * G[1][1] - G[1][0] * G[0][1]) * invdet;

     PetscReal localWidth = 2.0*sqrt(Ginv[0][0]+Ginv[1][1]+Ginv[2][2]);

     PetscScalar phi  = u[0];
     PetscScalar dens = u_main[0];
     PetscScalar alpha_1 =  0.5*(tanh(phi/(localWidth))/0.987+1.0);
     PetscScalar alpha_1_reference = u_main[5];

     PetscReal gradPhiGgradPhi = 0.0;
     for (int i=0;i<dim;i++){
       for (int j=0;j<dim;j++){
         gradPhiGgradPhi += grad_u[0][i]*G[i][j]*grad_u[0][j];
       }
     }

     PetscReal phi_reference = (localWidth)*atanh(0.987*(2.0*alpha_1-1.0));
     PetscReal S             = regularizedSignFunction(user, phi_reference, localWidth);

       for(int i = 0 ; i < dim ; i ++){
         convVel[i] = S*(grad_u[0][i]/sqrt(grad_u[0][0]*grad_u[0][0]+grad_u[0][1]*grad_u[0][1]+grad_u[0][2]*grad_u[0][2]));
       }

       PetscReal dt_LS = par->timeStep/par->timeTildeFactor;
       PetscReal adotGa = 0.0;
       for(int i = 0 ; i < dim ; i ++){
         for(int j = 0 ; j < dim ; j ++){
           adotGa += convVel[i]*G[i][j]*convVel[j];
         }
       }

     PetscReal Tau_phi = 1.0/sqrt(4.0/(dt_LS*dt_LS)+adotGa);
     PetscInt   m;
     PetscReal    *N0 = pnt->shape[0];
     PetscReal   (*N1)[dim] = (PetscReal (*)[dim]) pnt->shape[1];
     PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
     PetscInt a,nen=pnt->nen;
     PetscReal   normGradPhi = sqrt(grad_u[0][0]*grad_u[0][0]+grad_u[0][1]*grad_u[0][1]+grad_u[0][2]*grad_u[0][2]);
     for (a=0; a<nen; a++) {
       R[a][0] = 0.0;
       if(!pnt->atboundary){
         R[a][0] += (N0[a] + Tau_phi*(convVel[0]*N1[a][0]+convVel[1]*N1[a][1]+convVel[2]*N1[a][2]))*
         (u_t[0]+S*(normGradPhi-1.0));}
         //Penalty Term for large density difference or pressure driven expansion flows
         R[a][0] += N0[a]*2.0*(phi-phi_reference)*(1.0 - tanh(phi/localWidth)*tanh(phi/localWidth))*0.5/(0.987*localWidth);
       }
   return 0;
 }

 #undef  __FUNCT__
 #define __FUNCT__ "Residual_LSC"
 PetscErrorCode Residual_LSC(IGAPoint pnt,
                         const PetscScalar *V,
                         const PetscScalar *U,
                         PetscScalar *Re,
                         const PetscScalar *U_main,
                         IGAPoint pnt_main,
                         const PetscScalar *U0_LS,
                         const PetscScalar *U_LS,
                         IGAPoint pnt_LS,
                         PARAMETERS *par,
                         AppCtx *user)
 {
     PetscErrorCode ierr;

     PetscInt i,j,l;
     PetscInt  dof   = pnt->dof;
     PetscInt  dim   = pnt->dim;
     PetscScalar u[dof], u_t[dof], grad_u[dof][dim], u_LS[1], u0_LS[1];
     PetscReal G[3][3] = {{0.0}};
     PetscReal InvGradMap[dim][dim];
     PetscScalar u_main[6];

     IGAPointFormValue(pnt,U,&u[0]);
     IGAPointFormValue(pnt,V,&u_t[0]);
     IGAPointFormGrad (pnt,U,&grad_u[0][0]);
     IGAPointFormValue(pnt_main,U_main,&u_main[0]);
     IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);
     IGAPointFormValue(pnt_LS,U_LS,&u_LS[0]);
     IGAPointFormValue(pnt_LS,U0_LS,&u0_LS[0]);

     for (i=0;i<dim;i++){
       for (j=0;j<dim;j++){
         for (l=0;l<dim;l++){
           G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
         }
       }
     }
     PetscReal Ginv[3][3]={{0.0}};
     PetscReal det = G[0][0] * (G[1][1] * G[2][2] - G[2][1] * G[1][2]) -
                     G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) +
                     G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);

     PetscReal invdet = 1.0 / det;

     Ginv[0][0] = (G[1][1] * G[2][2] - G[2][1] * G[1][2]) * invdet;
     Ginv[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) * invdet;
     Ginv[2][2] = (G[0][0] * G[1][1] - G[1][0] * G[0][1]) * invdet;

     PetscReal localWidth = 2.0*sqrt(Ginv[0][0]+Ginv[1][1]+Ginv[2][2]);

     PetscReal   *N0        = pnt->shape[0];
     PetscScalar phi_prime  = u[0];
     PetscScalar phi        = u_LS[0];
     PetscScalar phi0       = u0_LS[0];
     PetscScalar dens       = u_main[0];
     PetscReal   phi_tilde  = phi + phi_prime;
     PetscReal   h          = localWidth;

     PetscScalar alpha_1_reference = 0.5*(tanh(phi0/h)/0.987+1.0);

     PetscReal   (*N1)[dim] = (PetscReal (*)[dim]) pnt_main->shape[1];
     PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
     PetscInt a,nen=pnt->nen;

     for (a=0; a<nen; a++) {
       R[a][0] = 0.0;
       if(!pnt->atboundary){
         R[a][0] += N0[a]*(alpha(user, phi_tilde, localWidth) - alpha_1_reference);
         if(((1.0-tanh(phi_tilde/h)*tanh(phi_tilde/h))/h)<1.0e-10){R[a][0] = 0.0;}
         // Penalize devaition from constant
         //R[a][0] += 10.0*h*(grad_u[0][0]*N1[a][0]+grad_u[0][1]*N1[a][1]);
       }
     }

   return 0;
 }
//// End Residual (Fluid) Definitions ////


//// Tangent Integrands ////
#undef  __FUNCT__
#define __FUNCT__ "Tangent"
PetscErrorCode Tangent(IGAPoint pnt,
                       PetscReal shift,const PetscScalar *V,
                       PetscReal t,const PetscScalar *U,
                       PetscScalar *Ke,void *ctx)
{
// Tangent matrix KK_{ajal} = A0_{jl}*shift*N_a, immersing fluid
  PetscErrorCode ierr;
  AppCtx      *user = (AppCtx *)ctx;
  PetscInt 	  dof   = pnt->dof;
  PetscInt    dim   = pnt->dim;
  PetscReal   Cv    = user->Cv;
  PetscScalar u[dof], grad_u[dof][dim];
  PetscReal G[3][3] = {{0.0}};
  PetscReal InvGradMap[dim][dim];

  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);
  for (int i=0;i<dim;i++){
    for (int j=0;j<dim;j++){
      for (int l=0;l<dim;l++){
        G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
      }
    }
  }

  PetscReal gradPhiGgradPhi = 0.0;
  for (int i=0;i<dim;i++){
    for (int j=0;j<dim;j++){
      gradPhiGgradPhi += grad_u[5][i]*G[i][j]*grad_u[5][j];
    }
  }

  PetscReal localWidth = 0.5*sqrt(grad_u[5][0]*grad_u[5][0]+grad_u[5][1]*grad_u[5][1]+grad_u[5][2]*grad_u[5][2])/sqrt(gradPhiGgradPhi);

  PetscReal *N0 = pnt->shape[0];
  PetscReal  A0[6][6]={{0.0}};

  PetscScalar dens    = u[0];
  PetscScalar ux      = u[1];
  PetscScalar uy      = u[2];
  PetscScalar uz      = u[3];
  PetscScalar temp    = u[4];
  PetscScalar alpha_1 = u[5];
  PetscScalar alpha_2 = 1.0 - alpha_1;

  //// Fluid Property reconstruction
  PetscReal P_1, fprime_1, cs_1, energy_1;
  PetscReal P_2, fprime_2, cs_2, energy_2;

  P_2      = TaitEq_WATER(dens, &fprime_2, &cs_2);
  energy_2 = Energy_WATER(temp);
  P_1      = JWL_RDX(dens, &fprime_1, &cs_1);
  energy_1 = Energy_RDX(temp);

  PetscReal P, fprime, cs, e;
  P      = alpha_1*P_1      + alpha_2*P_2;
  e      = alpha_1*energy_1 + alpha_2*energy_2;
  ////////////////////////////////////

  // Density Primitive A0: Conservation of Species Mass
  A0[0][0] = 1.0;
  A0[1][0] = ux;
  A0[1][1] = dens;
  A0[2][0] = uy;
  A0[2][2] = dens;
  A0[3][0] = uz;
  A0[3][3] = dens;
  A0[4][0] = e;
  A0[4][4] = dens*e/temp;
  A0[4][5] = dens*(energy_1-energy_2);
  A0[5][0] = alpha_1;
  A0[5][5] = dens;

  PetscScalar (*KK)[dof][pnt->nen][dof] = (PetscScalar (*)[dof][pnt->nen][dof])Ke;
  for (int a=0 ; a<pnt->nen ; a++) { //Only un-comment nonzero terms in A0 to save computational time
      KK[a][0][a][0] = A0[0][0]*shift*N0[a];
      //KK[a][0][a][1] = A0[0][1]*shift*N0[a];
      //KK[a][0][a][2] = A0[0][2]*shift*N0[a];
      //KK[a][0][a][3] = A0[0][3]*shift*N0[a];
      //KK[a][0][a][4] = A0[0][4]*shift*N0[a];
      //KK[a][0][a][5] = A0[0][5]*shift*N0[a];

      KK[a][1][a][0] = A0[1][0]*shift*N0[a];
      KK[a][1][a][1] = A0[1][1]*shift*N0[a];
      // KK[a][1][a][2] = A0[1][2]*shift*N0[a];
      // KK[a][1][a][3] = A0[1][3]*shift*N0[a];
      // KK[a][1][a][4] = A0[1][4]*shift*N0[a];
      // KK[a][1][a][5] = A0[1][5]*shift*N0[a];

      KK[a][2][a][0] = A0[2][0]*shift*N0[a];
      // KK[a][2][a][1] = A0[2][1]*shift*N0[a];
      KK[a][2][a][2] = A0[2][2]*shift*N0[a];
      // KK[a][2][a][3] = A0[2][3]*shift*N0[a];
      // KK[a][2][a][4] = A0[2][4]*shift*N0[a];
      // KK[a][2][a][5] = A0[2][5]*shift*N0[a];

      KK[a][3][a][0] = A0[3][0]*shift*N0[a];
      // KK[a][3][a][1] = A0[3][1]*shift*N0[a];
      // KK[a][3][a][2] = A0[3][2]*shift*N0[a];
      KK[a][3][a][3] = A0[3][3]*shift*N0[a];
      // KK[a][3][a][4] = A0[3][4]*shift*N0[a];
      // KK[a][3][a][5] = A0[3][5]*shift*N0[a];

      KK[a][4][a][0] = A0[4][0]*shift*N0[a];
      //KK[a][4][a][1] = A0[4][1]*shift*N0[a];
      //KK[a][4][a][2] = A0[4][2]*shift*N0[a];
      //KK[a][4][a][3] = A0[4][3]*shift*N0[a];
      KK[a][4][a][4] = A0[4][4]*shift*N0[a];
      KK[a][4][a][5] = A0[4][5]*shift*N0[a];

      KK[a][5][a][0] = A0[5][0]*shift*N0[a];
      // KK[a][5][a][1] = A0[5][1]*shift*N0[a];
      // KK[a][5][a][2] = A0[5][2]*shift*N0[a];
      // KK[a][5][a][3] = A0[5][3]*shift*N0[a];
      // KK[a][5][a][4] = A0[5][4]*shift*N0[a];
      KK[a][5][a][5] = A0[5][5]*shift*N0[a];
      //Debugging
      // for(int i = 0 ; i < 6 ; i++){
      //   for(int j = 0 ; j < 6 ; j++){
      //
      //     if(KK[a][i][a][j]!=KK[a][i][a][j]){
      //       PetscPrintf(PETSC_COMM_SELF,"NaN detected in Mass Matrix at %d %d, exiting...\n", i, j);
      //       exit(1);
      //     }
      //   }
      // }
  }
  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent_LS"
PetscErrorCode Tangent_LS(IGAPoint pnt,
                       PetscReal shift,
                       const PetscScalar *V,
                       PetscReal t,
                       const PetscScalar *U,
                       PetscScalar *Ke,
                       void *ctx,
                       const PetscScalar *U_main,
                       IGAPoint pnt_main,
                       PARAMETERS *par)
{
  PetscErrorCode ierr;

  AppCtx *user = (AppCtx *)ctx;
  PetscInt i,j,l;
  PetscInt 	dof = pnt->dof;
  PetscInt  dim = pnt->dim;

  PetscScalar u[dof], grad_u[dof][dim], u_t[dof];
  PetscScalar u_main[6];
  PetscReal   InvGradMap[dim][dim];
  PetscReal   G[3][3] = {{0.0}};

  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  IGAPointFormValue(pnt,V,&u_t[0]);
  IGAPointFormValue(pnt_main,U_main,&u_main[0]);
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);

  PetscReal   *N0 = pnt->shape[0];
  PetscReal   (*N1)[dim] = (PetscReal (*)[dim]) pnt->shape[1];
  PetscReal   convVel[2] = {0.0};
  PetscScalar phi  = u[0];
  PetscScalar alpha_1 = u_main[5];
  PetscReal   gradPhiGgradPhi = 0.0;
  if(alpha_1>1.0){alpha_1 = 1.0;}
  if(alpha_1<0.0){alpha_1 = 0.0;}

  for (i=0;i<dim;i++){
    for (j=0;j<dim;j++){
      for (l=0;l<dim;l++){
        G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
      }
    }
  }
  PetscReal Ginv[3][3]={{0.0}};
  PetscReal det = G[0][0] * (G[1][1] * G[2][2] - G[2][1] * G[1][2]) -
                  G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) +
                  G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);

  PetscReal invdet = 1.0 / det;

  Ginv[0][0] = (G[1][1] * G[2][2] - G[2][1] * G[1][2]) * invdet;
  Ginv[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) * invdet;
  Ginv[2][2] = (G[0][0] * G[1][1] - G[1][0] * G[0][1]) * invdet;

  PetscReal localWidth = 2.0*sqrt(Ginv[0][0]+Ginv[1][1]+Ginv[2][2]);

  for (int i=0;i<dim;i++){
    for (int j=0;j<dim;j++){
      gradPhiGgradPhi += grad_u[0][i]*G[i][j]*grad_u[0][j];
    }
  }


  PetscReal phi_reference = (localWidth)*atanh(0.987*(2.0*alpha_1-1.0));
  PetscReal S             = regularizedSignFunction(user, phi_reference, localWidth);

  for(int i = 0 ; i < dim ; i++){
    convVel[i] = S*grad_u[0][i]/sqrt(grad_u[0][0]*grad_u[0][0]+grad_u[0][1]*grad_u[0][1]+grad_u[0][2]*grad_u[0][2]);
  }

  PetscReal dt = par->timeStep/par->timeTildeFactor;
  PetscReal adotGa = 0.0;
  for(int i = 0 ; i < dim ; i++){
    for(int j = 0 ; j < dim ; j++){
      adotGa += convVel[i]*G[i][j]*convVel[j];
    }
  }

  PetscReal Tau_phi = 1.0/sqrt(4.0/(dt*dt)+adotGa);
  PetscInt  nen     = pnt->nen;

  PetscScalar (*KK)[dof][nen][dof] = (PetscScalar (*)[dof][nen][dof])Ke;
  for(int a = 0 ; a<nen ; a++){
    for(int b = 0 ; b<nen ; b++){
     KK[a][0][b][0] += N0[b]*(N0[a]+(convVel[0]*N1[a][0]+convVel[1]*N1[a][1]+convVel[2]*N1[a][2])*Tau_phi)*shift;
    }
  }

  return 0;
}

#undef  __FUNCT__
#define __FUNCT__ "Tangent_LSC"
PetscErrorCode Tangent_LSC(IGAPoint pnt,
                       const PetscScalar *V,
                       const PetscScalar *U,
                       PetscScalar *Ke,
                       void *ctx,
                       const PetscScalar *U_main,
                       IGAPoint pnt_main,
                       const PetscScalar *U_LS,
                       IGAPoint pnt_LS,
                       PARAMETERS *par)
{
  AppCtx *user = (AppCtx *)ctx;
  PetscInt i,j,l;
  PetscInt 	dof = pnt->dof;
  PetscInt  dim = pnt->dim;

  PetscScalar u[dof], u_LS[1], u_main[6];
  PetscReal InvGradMap[dim][dim];
  PetscReal G[3][3] = {{0.0}};

  IGAPointFormValue(pnt,U,&u[0]);
  IGAPointFormValue(pnt_main,U_main,&u_main[0]);
  IGAPointFormValue(pnt_LS,U_LS,&u_LS[0]);
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);

  for (i=0;i<dim;i++){
    for (j=0;j<dim;j++){
      for (l=0;l<dim;l++){
        G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
      }
    }
  }
  PetscReal Ginv[3][3]={{0.0}};
  PetscReal det = G[0][0] * (G[1][1] * G[2][2] - G[2][1] * G[1][2]) -
                  G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) +
                  G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);

  PetscReal invdet = 1.0 / det;

  Ginv[0][0] = (G[1][1] * G[2][2] - G[2][1] * G[1][2]) * invdet;
  Ginv[1][1] = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) * invdet;
  Ginv[2][2] = (G[0][0] * G[1][1] - G[1][0] * G[0][1]) * invdet;

  PetscReal localWidth = 2.0*sqrt(Ginv[0][0]+Ginv[1][1]+Ginv[2][2]);

  PetscReal   *N0        = pnt->shape[0];
  PetscScalar phi_prime  = u[0];
  PetscScalar dens       = u_main[0];
  PetscScalar phi        = u_LS[0];

  PetscReal phi_tilde    = phi + phi_prime;

  PetscInt    a,nen=pnt->nen;
  PetscReal   h = localWidth;
  PetscScalar (*KK)[dof][nen][dof] = (PetscScalar (*)[dof][nen][dof])Ke;
  for(int a = 0 ; a<nen ; a++){
    for(int b = 0 ; b<nen ; b++){
     KK[a][0][b][0] += N0[b]*N0[a]*0.5*(1.0-tanh(phi_tilde/h)*tanh(phi_tilde/h))/(h*0.987);
     if(((1.0-tanh(phi_tilde/h)*tanh(phi_tilde/h))/h)<1.0e-10){KK[a][0][a][0] = 1.0;
     if(a!=b){KK[a][0][b][0] = 0.0;}}

   }
  }
  return 0;
}

//////////////////////////

//// Integration functions ////
#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIJacobianComp"
PetscErrorCode IGAComputeIJacobianComp(IGA iga,
                                       PetscReal a,Vec vecV,
                                       PetscReal t,Vec vecU,
                                       Mat matJ,AppCtx *user)
{
  Vec               localV;
  Vec               localU;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  IGAElement        element;
  IGAPoint          point;
  IGAFormIJacobian  IJacobian;
  PetscScalar       *V,*U,*J,*K;
  PetscErrorCode    ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);
  /*ierr = IGASetUserIJacobian(iga,IJacobian,&user);CHKERRQ(ierr);*/
  ierr = IGASetFormIJacobian(iga,IJacobian,&user);CHKERRQ(ierr);
  /*IGACheckFormOp(iga,1,IJacobian);*/

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
//  PetscPrintf(PETSC_COMM_WORLD, "About to get local/global vec array\n");
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
//  PetscPrintf(PETSC_COMM_WORLD, "done\n");

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  while (IGANextElement(iga,element)) {
    /*ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);*/
    /*ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);*/
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
//    PetscPrintf(PETSC_COMM_WORLD, "got work mat\n");
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
  //  PetscPrintf(PETSC_COMM_WORLD, "got values\n");
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);

    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    while (IGAElementNextPoint(element,point)) {
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = Tangent(point,a,V,t,U,K,user);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }
    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }

  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = IGAPointDestroy(&point);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIJacobianComp_LS"
PetscErrorCode IGAComputeIJacobianComp_LS(IGA iga,
                                       PetscReal a,
                                       Vec vecV,
                                       PetscReal t,
                                       Vec vecU,
                                       IGA iga_main,
                                       Vec vecV_main,
                                       Vec vecU_main,
                                       Mat matJ,
                                       AppCtx *user,
                                       PARAMETERS *par)
{
  Vec               localV, localV_main;
  Vec               localU, localU_main;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayV_main;
  const PetscScalar *arrayU_main;
  IGAElement        element, element_main;
  IGAPoint          point;
  IGAFormIJacobian  IJacobian;
  void              *ctx;
  PetscScalar       *V,*U,*J,*K;
  PetscScalar       *V_main,*U_main,*J_main;
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);

  /*ierr = IGASetUserIJacobian(iga,IJacobian,&user);CHKERRQ(ierr);*/
  ierr = IGASetFormIJacobian(iga,IJacobian,&user);CHKERRQ(ierr);
  /*IGACheckFormOp(iga,1,IJacobian);*/

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = IGAGetLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_main,&element_main);CHKERRQ(ierr);

  while (IGANextElement(iga,element) && IGANextElement(iga_main, element_main)) {
    IGAPoint point_main;
    /*ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);*/
    /*ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);*/
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);

    ierr = IGAElementGetValues(element_main,arrayU_main,&U_main);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_main,U_main);CHKERRQ(ierr);

    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAElementBeginPoint(element_main,&point_main);CHKERRQ(ierr);

    while (IGAElementNextPoint(element,point) && IGAElementNextPoint(element_main,point_main)) {
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = Tangent_LS(point, a, V, t, U, K, user, U_main, point_main, par);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }

    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }

  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIFunction_LS"
PetscErrorCode IGAComputeIFunction_LS(IGA iga,
                                   PetscReal a,Vec vecV,
                                   PetscReal t,Vec vecU,
                                   IGA iga_main,
                                   Vec vecV_main,
                                   Vec vecU_main,
                                   Vec vecF,
                                   PARAMETERS *par,
                                   AppCtx *user)
{

  Vec               localV, localV_main;
  Vec               localU, localU_main;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayV_main;
  const PetscScalar *arrayU_main;
  IGAElement        element, element_main;
  IGAPoint          point;
  IGAFormIFunction  IFunction;
  void              *ctx;
  PetscScalar       *V,*U,*F,*R;
  PetscScalar       *V_main,*U_main,*F_main;
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,IFunction);

  /* Clear global vector F */
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = IGAGetLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_main,&element_main);CHKERRQ(ierr);
  while (IGANextElement(iga,element) && IGANextElement(iga_main, element_main)) {
    IGAPoint point_main;

    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementDelValues(element,V);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);

    ierr = IGAElementGetValues(element_main,arrayU_main,&U_main);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_main,U_main);CHKERRQ(ierr);

    /* FormIFunction loop */
    while (IGAElementNextFormIFunction(element,&IFunction,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      ierr = IGAElementBeginPoint(element_main,&point_main);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point) && IGAElementNextPoint(element_main,point_main)) {
        ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);
        ierr = Residual_LS(point,a,V,t,U,R, U_main, point_main, par, user);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Restore local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = IGARestoreLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIJacobianComp_LSC"
PetscErrorCode IGAComputeIJacobianComp_LSC(IGA iga,
                                       Vec vecV,
                                       Vec vecU,
                                       IGA iga_main,
                                       Vec vecV_main,
                                       Vec vecU_main,
                                       IGA iga_LS,
                                       Vec vecV_LS,
                                       Vec vecU_LS,
                                       Mat matJ,
                                       AppCtx *user,
                                       PARAMETERS *par)
{
  Vec               localV, localV_main, localV_LS;
  Vec               localU, localU_main, localU_LS;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayV_main;
  const PetscScalar *arrayU_main;
  const PetscScalar *arrayV_LS;
  const PetscScalar *arrayU_LS;
  IGAElement        element, element_main, element_LS;
  IGAPoint          point;
  IGAFormIJacobian  IJacobian;
  void              *ctx;
  PetscScalar       *V,*U,*J,*K;
  PetscScalar       *V_main,*U_main,*J_main;
    PetscScalar     *V_LS,*U_LS,*J_LS;
  PetscErrorCode    ierr;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(matJ,MAT_CLASSID,7);
  IGACheckSetUp(iga,1);

  /*ierr = IGASetUserIJacobian(iga,IJacobian,&user);CHKERRQ(ierr);*/
  ierr = IGASetFormIJacobian(iga,IJacobian,&user);CHKERRQ(ierr);
  /*IGACheckFormOp(iga,1,IJacobian);*/

  /* Clear global matrix J*/
  ierr = MatZeroEntries(matJ);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = IGAGetLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

  ierr = IGAGetLocalVecArray(iga_LS,vecV_LS,&localV_LS,&arrayV_LS);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga_LS,vecU_LS,&localU_LS,&arrayU_LS);CHKERRQ(ierr);

  /* Element Loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_main,&element_main);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_LS,&element_LS);CHKERRQ(ierr);

  while (IGANextElement(iga,element) && IGANextElement(iga_main, element_main) && IGANextElement(iga_LS, element_LS)) {
    IGAPoint point_main;
    IGAPoint point_LS;
    /*ierr = IGAElementGetWorkVal(element,&V);CHKERRQ(ierr);*/
    /*ierr = IGAElementGetWorkVal(element,&U);CHKERRQ(ierr);*/
    ierr = IGAElementGetWorkMat(element,&J);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);

    ierr = IGAElementGetValues(element_main,arrayU_main,&U_main);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_main,U_main);CHKERRQ(ierr);

    ierr = IGAElementGetValues(element_LS,arrayU_LS,&U_LS);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_LS,U_LS);CHKERRQ(ierr);

    /* Quadrature loop */
    ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAElementBeginPoint(element_main,&point_main);CHKERRQ(ierr);
    ierr = IGAElementBeginPoint(element_LS,&point_LS);CHKERRQ(ierr);

    while (IGAElementNextPoint(element,point) && IGAElementNextPoint(element_main,point_main) && IGAElementNextPoint(element_LS,point_LS)) {
      ierr = IGAPointGetWorkMat(point,&K);CHKERRQ(ierr);
      ierr = Tangent_LSC(point, V, U, K, user, U_main, point_main, U_LS, point_LS, par);CHKERRQ(ierr);
      ierr = IGAPointAddMat(point,K,J);CHKERRQ(ierr);
    }

    ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    ierr = IGAElementFixJacobian(element,J);CHKERRQ(ierr);
    ierr = IGAElementAssembleMat(element,J,matJ);CHKERRQ(ierr);
  }

  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  /* Get local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_LS,vecV_LS,&localV_LS,&arrayV_LS);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_LS,vecU_LS,&localU_LS,&arrayU_LS);CHKERRQ(ierr);

  /* Assemble global matrix J*/
  ierr = MatAssemblyBegin(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(matJ,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "IGAComputeIFunction_LSC"
PetscErrorCode IGAComputeIFunction_LSC(IGA iga,
                                   Vec vecV,
                                   Vec vecU,
                                   IGA iga_main,
                                   Vec vecV_main,
                                   Vec vecU_main,
                                   IGA iga_LS,
                                   Vec vecV_LS, //This stores the reference phi values
                                   Vec vecU_LS,
                                   Vec vecF,
                                   PARAMETERS *par,
                                   AppCtx *user)
{

  Vec               localV, localV_LS, localV_main;
  Vec               localU, localU_LS, localU_main;
  const PetscScalar *arrayV;
  const PetscScalar *arrayU;
  const PetscScalar *arrayV_main;
  const PetscScalar *arrayU_main;
  const PetscScalar *arrayV_LS;
  const PetscScalar *arrayU_LS;
  IGAElement        element, element_main, element_LS;
  IGAPoint          point;
  IGAFormIFunction  IFunction;
  PetscScalar       *V,*U,*F,*R;
  PetscScalar       *V_main,*U_main,*F_main;
  PetscScalar       *V_LS,*U_LS,*F_LS;
  PetscErrorCode    ierr;
  void              *ctx;
  PetscFunctionBegin;

  PetscValidHeaderSpecific(iga,IGA_CLASSID,1);
  PetscValidHeaderSpecific(vecV,VEC_CLASSID,4);
  PetscValidHeaderSpecific(vecU,VEC_CLASSID,6);
  PetscValidHeaderSpecific(vecF,VEC_CLASSID,7);
  IGACheckSetUp(iga,1);
  IGACheckFormOp(iga,1,IFunction);

  /* Get local vectors V,U and arrays */
  ierr = IGAGetLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = IGAGetLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGAGetLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

    ierr = IGAGetLocalVecArray(iga_LS,vecV_LS,&localV_LS,&arrayV_LS);CHKERRQ(ierr);
    ierr = IGAGetLocalVecArray(iga_LS,vecU_LS,&localU_LS,&arrayU_LS);CHKERRQ(ierr);

  ierr = PetscLogEventBegin(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Element loop */
  ierr = IGABeginElement(iga,&element);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_main,&element_main);CHKERRQ(ierr);
  ierr = IGABeginElement(iga_LS,&element_LS);CHKERRQ(ierr);
  while (IGANextElement(iga,element) && IGANextElement(iga_LS,element_LS) && IGANextElement(iga_main, element_main)) {
    IGAPoint point_main;
    IGAPoint point_LS;

    ierr = IGAElementGetWorkVec(element,&F);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayV,&V);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element,arrayU,&U);CHKERRQ(ierr);
    ierr = IGAElementDelValues(element,V);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element,U);CHKERRQ(ierr);

    ierr = IGAElementGetValues(element_LS,arrayU_LS,&U_LS);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_LS,U_LS);CHKERRQ(ierr);
    ierr = IGAElementGetValues(element_LS,arrayV_LS,&V_LS);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_LS,V_LS);CHKERRQ(ierr);

    ierr = IGAElementGetValues(element_main,arrayU_main,&U_main);CHKERRQ(ierr);
    ierr = IGAElementFixValues(element_main,U_main);CHKERRQ(ierr);

    /* FormIFunction loop */
    while (IGAElementNextFormIFunction(element,&IFunction,&ctx)) {
      /* Quadrature loop */
      ierr = IGAElementBeginPoint(element,&point);CHKERRQ(ierr);
      ierr = IGAElementBeginPoint(element_LS,&point_LS);CHKERRQ(ierr);
      ierr = IGAElementBeginPoint(element_main,&point_main);CHKERRQ(ierr);
      while (IGAElementNextPoint(element,point) && IGAElementNextPoint(element_LS,point_LS) && IGAElementNextPoint(element_main,point_main)) {
        ierr = IGAPointGetWorkVec(point,&R);CHKERRQ(ierr);

        ierr = Residual_LSC(point, V, U, R, U_main, point_main, V_LS, U_LS, point_LS, par, user);CHKERRQ(ierr);
        ierr = IGAPointAddVec(point,R,F);CHKERRQ(ierr);
      }
      ierr = IGAElementEndPoint(element,&point);CHKERRQ(ierr);
    }
    ierr = IGAElementFixFunction(element,F);CHKERRQ(ierr);
    ierr = IGAElementAssembleVec(element,F,vecF);CHKERRQ(ierr);
  }
  ierr = IGAEndElement(iga,&element);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(IGA_FormIFunction,iga,vecV,vecU,vecF);CHKERRQ(ierr);

  /* Restore local vectors V,U and arrays */
  ierr = IGARestoreLocalVecArray(iga,vecV,&localV,&arrayV);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,vecU,&localU,&arrayU);CHKERRQ(ierr);

  ierr = IGARestoreLocalVecArray(iga_LS,vecV_LS,&localV_LS,&arrayV_LS);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_LS,vecU_LS,&localU_LS,&arrayU_LS);CHKERRQ(ierr);

  ierr = IGARestoreLocalVecArray(iga_main,vecV_main,&localV_main,&arrayV_main);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga_main,vecU_main,&localU_main,&arrayU_main);CHKERRQ(ierr);

  /* Assemble global vector F */
  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
///////////////////////////////

//// Generalized alpha Time Integration + Lagrangian update routines ////
#undef __FUNCT__
#define __FUNCT__ "TSPredictStage_GeneralizedAlpha_LS"
static PetscErrorCode TSPredictStage_GeneralizedAlpha_LS(AppCtx *user,
                                                      PARAMETERS *par)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec V0 = user->V0_LS, A0 = user->A0_LS, D0 = user->D0_LS;
  Vec Vp = user->Vp_LS, Ap = user->Ap_LS, Dp = user->Dp_LS;

  PetscReal dt = par->timeStep/par->timeTildeFactor;
  PetscReal Gamma = user->Gamma;
  PetscReal Beta = user->Beta;



  //FOR FLUID MECHANICS AND FSI
  //Vp = V0
  ierr = VecCopy(V0,Vp);CHKERRQ(ierr);

  //Ap = (Gamma - 1)/Gamma*A0
  ierr = VecCopy(A0,Ap);CHKERRQ(ierr);
  ierr = VecScale(Ap,(Gamma - 1.0)/Gamma);CHKERRQ(ierr);

  //Xp = X0 + dt*V0 + dt²/2*((1-2*Beta)*A0 + 2*Beta*Ap)
  // Dp = D0 +dt*V0 +dt*dt/2*((1-2*Beta)*A0 + 2*Beta*Ap);
  ierr = VecAXPBYPCZ(Dp,dt,dt*dt/2*(1-2*Beta),0,V0,A0);
  ierr = VecAXPBYPCZ(Dp,1,dt*dt*Beta,1,D0,Ap);

  /*
  //FORN NONLINEAR SOLID MECHANICS
  // Xp = X
  ierr = VecCopy(X0,Xp);CHKERRQ(ierr);

  // Ap = - 1/(Beta*dt)*V0 - (1-2*Beta)/(2*Beta)*A0
  ierr = VecCopy(V0,Ap);CHKERRQ(ierr);  //Ap=V0
  ierr = VecAXPBY(Ap,-(1-2*Beta)/(2*Beta),-1/(dt*Beta),A0);CHKERRQ(ierr); //Ap= - 1/(Beta*dt)*Ap - (1-2*Beta)/(2*Beta)*A0


  // Vp = V0 + dt*((1-Gamma)*A0 + Gamma*Ap)
  ierr = VecWAXPY(Vp,(1.0-Gamma)/Gamma,A0,Ap);CHKERRQ(ierr); //Vp = (1.0-Gamma)/Gamma*A0 + Ap
  ierr = VecAYPX (Vp,dt*Gamma,V0);CHKERRQ(ierr); 			 //Vp = V0 + Gamma*dt*Vp
  */

  ierr = VecCopy(user->Vp_LS,user->V1_LS);CHKERRQ(ierr);
  ierr = VecCopy(user->Ap_LS,user->A1_LS);CHKERRQ(ierr);
  ierr = VecCopy(user->Dp_LS,user->D1_LS);CHKERRQ(ierr);
  /*
  PetscScalar *xx, *xy, *xz;
  ierr  = VecGetArray(th->X1,&xx);CHKERRQ(ierr);
  ierr  = VecGetArray(th->V1,&xy);CHKERRQ(ierr);
  ierr  = VecGetArray(th->A1,&xz);CHKERRQ(ierr);
  PetscInt k,i;
  for(i=0;i<10;i++) {
  for(k=0;k<5;k++) {
  PetscPrintf(PETSC_COMM_WORLD,"i: %d, Xp: %e, Vp: %e, Ap: %e \n",i, xx[i*5+k], xy[i*5+k], xz[i*5+k]);
  }
  }
  */
ierr = 0;
PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "TSUpdateStage_GeneralizedAlpha"
static PetscErrorCode TSUpdateStage_GeneralizedAlpha(AppCtx *user,PARAMETERS *par,Vec dA)
{

  Vec            V1 = user->V1, A1 = user->A1;
  PetscReal      dt = par->timeStep;
  PetscReal      Gamma   = user->Gamma;
  PetscErrorCode ierr;
  PetscFunctionBegin;

 //USING dA
    //A1 = A1 + dA
    ierr = VecAXPY(A1,-1.0,dA);CHKERRQ(ierr);  			//ok
    //V1 = V1 + Gamma*dt*dA
    ierr = VecAXPY(V1,-Gamma*dt,dA);CHKERRQ(ierr);			//ok
    //X1 = X1 + Beta*dt²*dA
    //ierr = VecAXPY (D1,-Beta*dt*dt,dA);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSUpdateAlphaLevels_GeneralizedAlpha"
static PetscErrorCode TSUpdateAlphaLevels_GeneralizedAlpha(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

  Vec            V1 = user->V1, A1 = user->A1, D1 = user->D1;
  Vec            Va = user->Va, Aa = user->Aa, Da = user->Da;
  Vec            V0 = user->V0, A0 = user->A0, D0 = user->D0;
  PetscReal      Alpha_m = user->Alpha_m;
  PetscReal      Alpha_f = user->Alpha_f;

  /* Va = V0 + Alpha_f*(V1-V0) */
  ierr = VecWAXPY(Va,-1.0,V0,V1);CHKERRQ(ierr);
  ierr = VecAYPX (Va,Alpha_f,V0);CHKERRQ(ierr);		//ok
  /* Aa = A0 + Alpha_m*(A1-A0) */
  ierr = VecWAXPY(Aa,-1.0,A0,A1);CHKERRQ(ierr);
  ierr = VecAYPX (Aa,Alpha_m,A0);CHKERRQ(ierr);		//ok

  ierr = VecWAXPY(Da,-1.0,D0,D1);CHKERRQ(ierr);
  ierr = VecAYPX (Da,Alpha_f,D0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSPredictStage_GeneralizedAlpha"
static PetscErrorCode TSPredictStage_GeneralizedAlpha(AppCtx *user,PARAMETERS *par)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;

    Vec            V0 = user->V0, A0 = user->A0, D0 = user->D0;
    Vec            Vp = user->Vp, Ap = user->Ap, Dp = user->Dp;

    PetscReal      dt = par->timeStep;
    PetscReal      Gamma   = user->Gamma;
    PetscReal      Beta    = user->Beta;

    //FOR FLUID MECHANICS AND FSI
    //Vp = V0
    ierr = VecCopy(V0,Vp);CHKERRQ(ierr);     						//ok

    //Ap = (Gamma - 1)/Gamma*A0
    ierr = VecCopy(A0,Ap);CHKERRQ(ierr);
    ierr = VecScale(Ap,(Gamma - 1.0)/Gamma);CHKERRQ(ierr);		//ok

    //Xp = X0 + dt*V0 + dt²/2*((1-2*Beta)*A0 + 2*Beta*Ap)
    // Dp = D0 +dt*V0 +dt*dt/2*((1-2*Beta)*A0 + 2*Beta*Ap);
    ierr = VecAXPBYPCZ(Dp,dt,dt*dt/2*(1-2*Beta),0,V0,A0);
    ierr = VecAXPBYPCZ(Dp,1,dt*dt*Beta,1,D0,Ap);

    /*
     //FOR NONLINEAR SOLID MECHANICS
     // Xp = X
     ierr = VecCopy(X0,Xp);CHKERRQ(ierr);

     // Ap = - 1/(Beta*dt)*V0 - (1-2*Beta)/(2*Beta)*A0
     ierr = VecCopy(V0,Ap);CHKERRQ(ierr);  //Ap=V0
     ierr = VecAXPBY(Ap,-(1-2*Beta)/(2*Beta),-1/(dt*Beta),A0);CHKERRQ(ierr); //Ap= - 1/(Beta*dt)*Ap - (1-2*Beta)/(2*Beta)*A0


     // Vp = V0 + dt*((1-Gamma)*A0 + Gamma*Ap)
     ierr = VecWAXPY(Vp,(1.0-Gamma)/Gamma,A0,Ap);CHKERRQ(ierr); //Vp = (1.0-Gamma)/Gamma*A0 + Ap
     ierr = VecAYPX (Vp,dt*Gamma,V0);CHKERRQ(ierr); 			 //Vp = V0 + Gamma*dt*Vp
     */

    ierr = VecCopy(user->Vp,user->V1);CHKERRQ(ierr);
    ierr = VecCopy(user->Ap,user->A1);CHKERRQ(ierr);
    ierr = VecCopy(user->Dp,user->D1);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetRadius_GeneralizedAlpha"
PetscErrorCode TSSetRadius_GeneralizedAlpha(AppCtx *user,PetscReal radius)
{
  PetscFunctionBegin;
  user->Alpha_m = (3.0-radius)/(2.0*(1.0 + radius));
  user->Alpha_f = 1/(1+radius);
  user->Gamma   = 0.5 + user->Alpha_m - user->Alpha_f;
  user->Beta    = 0.5 * (1 + user->Alpha_m - user->Alpha_f); user->Beta *= user->Beta;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "interpolateVelocityOnSolidNodes"
PetscErrorCode interpolateVelocityOnSolidNodes(PARAMETERS *par,
                                               AppCtx *user, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscReal uf[5] = {0.0};
  PetscReal pt[3] = {0.0};
  IGAProbe prb;
  ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);
  IGAProbeSetCollective(prb, PETSC_FALSE);
  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);

  for(auto it=its.first; it != its.second; ++it){
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);
    if(!info.isTask && fd.Inside==1 && fd.material!=0){
    pt[0] =  info.currentCoord[0]/user->Lx;
    pt[1] =  info.currentCoord[1]/user->Ly;
    pt[2] =  info.currentCoord[2]/user->Lz;
    ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
    ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

    fd.totalPhysicalVelocity[0] = uf[1];
    fd.totalPhysicalVelocity[1] = uf[2];
    fd.totalPhysicalVelocity[2] = uf[3];

    if(uf[0]<=0.0){
      PetscPrintf(PETSC_COMM_SELF,"Density < 0, Current Explosive Volume Update error: interpolateVelocityOnSolidNodes \n");
      exit(0);
    }

    }
    put(fd_property,v,fd);
  }
  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeVelocityGradient"
PetscErrorCode ComputeVelocityGradient(PARAMETERS *par, AppCtx *user,Vec vecU,const PetscScalar *arrayU,ParticleManager &manager)
{
  // Assign Velocity Gradient based on NURBS basis function gradients (not PD-VelGrad) for PD volume update
  // Reference Volume used to subtract foreground integrals over BG domain
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscReal pt[3];
  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);

  IGAProbe prb;
  IGAProbeCreate(user->iga,user->V1,&prb);
  IGAProbeSetCollective(prb, PETSC_FALSE);

  for(auto it=its.first; it != its.second; ++it){
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);
    PetscScalar grad_u[5][3];

    if((!info.isTask) && (fd.material==0) && (fd.Inside == 1)){

      pt[0] =  info.currentCoord[0]/user->Lx;
      pt[1] =  info.currentCoord[1]/user->Ly;
      pt[2] =  info.currentCoord[2]/user->Lz;
      ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
      ierr = IGAProbeFormGrad(prb, &grad_u[0][0]);CHKERRQ(ierr);

      fd.velocityGradient[0] = grad_u[1][0];
      fd.velocityGradient[1] = grad_u[1][1];
      fd.velocityGradient[2] = grad_u[1][2];
      fd.velocityGradient[3] = grad_u[2][0];
      fd.velocityGradient[4] = grad_u[2][1];
      fd.velocityGradient[5] = grad_u[2][2];

      fd.velocityGradient[6] = grad_u[3][0];
      fd.velocityGradient[7] = grad_u[3][1];
      fd.velocityGradient[8] = grad_u[3][2];

      put(fd_property,v,fd);
    }
  }
IGAProbeDestroy(&prb);
PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeDeformationGradient"
PetscErrorCode ComputeDeformationGradient(PARAMETERS *par,PetscReal af, ParticleManager &manager)
{
    // Compute alpha level det(Deformation Gradient) on FG-PD particles for integration volume update
    PetscErrorCode ierr;
    BGL_FORALL_VERTICES(v,manager.graph,Graph){
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);

    if((!info.isTask) && (fd.Inside == 1) && (fd.material==0)){

        PetscReal temp1[9] = {0.0};
        PetscReal temp2[9] = {0.0};
        PetscReal temp3;
        PetscReal inv[9]   = {0.0};
        PetscReal inter[9] = {0.0};
        PetscReal temp = 0.0;

        temp1[0] = 1.0 - par->timeStep*fd.velocityGradient[0]*af;
        temp1[1] = -par->timeStep*fd.velocityGradient[1]*af;
        temp1[2] = -par->timeStep*fd.velocityGradient[2]*af;
        temp1[3] = -par->timeStep*fd.velocityGradient[3]*af;
        temp1[4] = 1.0 - par->timeStep*fd.velocityGradient[4]*af;
        temp1[5] = -par->timeStep*fd.velocityGradient[5]*af;
        temp1[6] = -par->timeStep*fd.velocityGradient[6]*af;
        temp1[7] = -par->timeStep*fd.velocityGradient[7]*af;
        temp1[8] = 1.0 - par->timeStep*fd.velocityGradient[8]*af;

        temp2[0] = 1.0 + par->timeStep*fd.velocityGradient[0]*(1-af);
        temp2[1] = par->timeStep*fd.velocityGradient[1]*(1-af);
        temp2[2] = par->timeStep*fd.velocityGradient[2]*(1-af);
        temp2[3] = par->timeStep*fd.velocityGradient[3]*(1-af);
        temp2[4] = 1.0 + par->timeStep*fd.velocityGradient[4]*(1-af);
        temp2[5] = par->timeStep*fd.velocityGradient[5]*(1-af);
        temp2[6] = par->timeStep*fd.velocityGradient[6]*(1-af);
        temp2[7] = par->timeStep*fd.velocityGradient[7]*(1-af);
        temp2[8] = 1.0 + par->timeStep*fd.velocityGradient[8]*(1-af);

        temp = temp1[0]*(temp1[4]*temp1[8]-temp1[5]*temp1[7]) - temp1[1]*(temp1[3]*temp1[8]-temp1[5]*temp1[6]) + temp1[2]*(temp1[3]*temp1[7]-temp1[4]*temp1[6]);

        inv[0] = (temp1[8]*temp1[4]-temp1[7]*temp1[5])/temp;
        inv[1] = -(temp1[8]*temp1[1]-temp1[7]*temp1[2])/temp;
        inv[2] = (temp1[5]*temp1[1]-temp1[4]*temp1[2])/temp;
        inv[3] = -(temp1[8]*temp1[3]-temp1[6]*temp1[5])/temp;
        inv[4] = (temp1[8]*temp1[0]-temp1[6]*temp1[2])/temp;
        inv[5] = -(temp1[5]*temp1[0]-temp1[3]*temp1[2])/temp;
        inv[6] = (temp1[7]*temp1[3]-temp1[6]*temp1[4])/temp;
        inv[7] = -(temp1[7]*temp1[0]-temp1[6]*temp1[1])/temp;
        inv[8] = (temp1[4]*temp1[0]-temp1[3]*temp1[1])/temp;

        inter[0] = inv[0]*temp2[0] + inv[1]*temp2[3] + inv[2]*temp2[6];
        inter[1] = inv[0]*temp2[1] + inv[1]*temp2[4] + inv[2]*temp2[7];
        inter[2] = inv[0]*temp2[2] + inv[1]*temp2[5] + inv[2]*temp2[8];
        inter[3] = inv[3]*temp2[0] + inv[4]*temp2[3] + inv[5]*temp2[6];
        inter[4] = inv[3]*temp2[1] + inv[4]*temp2[4] + inv[5]*temp2[7];
        inter[5] = inv[3]*temp2[2] + inv[4]*temp2[5] + inv[5]*temp2[8];
        inter[6] = inv[6]*temp2[0] + inv[7]*temp2[3] + inv[8]*temp2[6];
        inter[7] = inv[6]*temp2[1] + inv[7]*temp2[4] + inv[8]*temp2[7];
        inter[8] = inv[6]*temp2[2] + inv[7]*temp2[5] + inv[8]*temp2[8];

        temp1[0] = inter[0]*fd.DeformationGradientOld[0] + inter[1]*fd.DeformationGradientOld[3] + inter[2]*fd.DeformationGradientOld[6];
        temp1[1] = inter[0]*fd.DeformationGradientOld[1] + inter[1]*fd.DeformationGradientOld[4] + inter[2]*fd.DeformationGradientOld[7];
        temp1[2] = inter[0]*fd.DeformationGradientOld[2] + inter[1]*fd.DeformationGradientOld[5] + inter[2]*fd.DeformationGradientOld[8];
        temp1[3] = inter[3]*fd.DeformationGradientOld[0] + inter[4]*fd.DeformationGradientOld[3] + inter[5]*fd.DeformationGradientOld[6];
        temp1[4] = inter[3]*fd.DeformationGradientOld[1] + inter[4]*fd.DeformationGradientOld[4] + inter[5]*fd.DeformationGradientOld[7];
        temp1[5] = inter[3]*fd.DeformationGradientOld[2] + inter[4]*fd.DeformationGradientOld[5] + inter[5]*fd.DeformationGradientOld[8];
        temp1[6] = inter[6]*fd.DeformationGradientOld[0] + inter[7]*fd.DeformationGradientOld[3] + inter[8]*fd.DeformationGradientOld[6];
        temp1[7] = inter[6]*fd.DeformationGradientOld[1] + inter[7]*fd.DeformationGradientOld[4] + inter[8]*fd.DeformationGradientOld[7];
        temp1[8] = inter[6]*fd.DeformationGradientOld[2] + inter[7]*fd.DeformationGradientOld[5] + inter[8]*fd.DeformationGradientOld[8];

        fd.alphaDeformationGradient[0] = fd.DeformationGradientOld[0] + af*(temp1[0] - fd.DeformationGradientOld[0]);
        fd.alphaDeformationGradient[1] = fd.DeformationGradientOld[1] + af*(temp1[1] - fd.DeformationGradientOld[1]);
        fd.alphaDeformationGradient[2] = fd.DeformationGradientOld[2] + af*(temp1[2] - fd.DeformationGradientOld[2]);
        fd.alphaDeformationGradient[3] = fd.DeformationGradientOld[3] + af*(temp1[3] - fd.DeformationGradientOld[3]);
        fd.alphaDeformationGradient[4] = fd.DeformationGradientOld[4] + af*(temp1[4] - fd.DeformationGradientOld[4]);
        fd.alphaDeformationGradient[5] = fd.DeformationGradientOld[5] + af*(temp1[5] - fd.DeformationGradientOld[5]);
        fd.alphaDeformationGradient[6] = fd.DeformationGradientOld[6] + af*(temp1[6] - fd.DeformationGradientOld[6]);
        fd.alphaDeformationGradient[7] = fd.DeformationGradientOld[7] + af*(temp1[7] - fd.DeformationGradientOld[7]);
        fd.alphaDeformationGradient[8] = fd.DeformationGradientOld[8] + af*(temp1[8] - fd.DeformationGradientOld[8]);

        fd.currentDeformationGradient[0] = temp1[0];
        fd.currentDeformationGradient[1] = temp1[1];
        fd.currentDeformationGradient[2] = temp1[2];
        fd.currentDeformationGradient[3] = temp1[3];
        fd.currentDeformationGradient[4] = temp1[4];
        fd.currentDeformationGradient[5] = temp1[5];
        fd.currentDeformationGradient[6] = temp1[6];
        fd.currentDeformationGradient[7] = temp1[7];
        fd.currentDeformationGradient[8] = temp1[8];

        temp3 = fd.currentDeformationGradient[0]*fd.currentDeformationGradient[8]*fd.currentDeformationGradient[4];
        temp3 += -fd.currentDeformationGradient[0]*fd.currentDeformationGradient[5]*fd.currentDeformationGradient[7];
        temp3 += -fd.currentDeformationGradient[1]*fd.currentDeformationGradient[3]*fd.currentDeformationGradient[8];
        temp3 += fd.currentDeformationGradient[1]*fd.currentDeformationGradient[5]*fd.currentDeformationGradient[6];
        temp3 += fd.currentDeformationGradient[2]*fd.currentDeformationGradient[3]*fd.currentDeformationGradient[7];
        temp3 += -fd.currentDeformationGradient[2]*fd.currentDeformationGradient[4]*fd.currentDeformationGradient[6];

        fd.determinantCurrentDeformationGradient = temp3;

        temp3 = fd.alphaDeformationGradient[0]*fd.alphaDeformationGradient[8]*fd.alphaDeformationGradient[4];
        temp3 += -fd.alphaDeformationGradient[0]*fd.alphaDeformationGradient[5]*fd.alphaDeformationGradient[7];
        temp3 += -fd.alphaDeformationGradient[1]*fd.alphaDeformationGradient[3]*fd.alphaDeformationGradient[8];
        temp3 += fd.alphaDeformationGradient[1]*fd.alphaDeformationGradient[5]*fd.alphaDeformationGradient[6];
        temp3 += fd.alphaDeformationGradient[2]*fd.alphaDeformationGradient[3]*fd.alphaDeformationGradient[7];
        temp3 += -fd.alphaDeformationGradient[2]*fd.alphaDeformationGradient[4]*fd.alphaDeformationGradient[6];

        fd.determinantAlphaDeformationGradient = temp3;
        put(fd_property,v,fd);
    }
    }
    PetscFunctionReturn(0);
}


/////// Functions for Penalty Coupling /////
#undef __FUNCT__
#define __FUNCT__ "computePenaltyOnSolid"
PetscErrorCode computePenaltyOnSolid(PARAMETERS *par,
                                     AppCtx *user, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;

  PetscInt i;
  PetscReal velDiff[3];
  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);

  for(auto it=its.first; it != its.second; ++it){
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);

    if(fd.material==0 && !info.isTask){
      for(i=0; i<user->iga->dim; i++){
        velDiff[i] = -fd.interpolatedVelocity[i] + fd.totalPhysicalVelocity[i];
      }

    for(i=0; i<user->iga->dim; i++){
      if(PETSC_TRUE){
        fd.residual[i] += fd.penaltyParameter *
        (velDiff[0]*fd.normal[0] + velDiff[1]*fd.normal[1] + velDiff[2]*fd.normal[2])*
        fd.normal[i];
      }
      else{
        fd.residual[i] = fd.penaltyParameter * velDiff[i];
      }
    }
  }
  put(fd_property,v,fd);
}

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "computePenaltyOnFluid"
PetscErrorCode computePenaltyOnFluid(PARAMETERS *par,
                                     AppCtx *user,
                                     Vec vecRes, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt i,a;
  PetscReal value = 0.0;
  PetscInt dim = user->iga->dim;
  PetscReal velDiff[dim];
  IGAProbe prb;
  PetscReal pt[3];
  PetscReal uf[5] = {0.0};

  ierr = VecZeroEntries(vecRes);CHKERRQ(ierr);
  ierr = IGAProbeCreate(user->iga,user->Va,&prb);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(prb, PETSC_FALSE);CHKERRQ(ierr);

  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){

    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);

    if((!info.isTask) && (fd.Inside == 1) && fd.material==0){
      pt[0] =  info.currentCoord[0]/user->Lx;
      pt[1] =  info.currentCoord[1]/user->Ly;
      pt[2] =  info.currentCoord[2]/user->Lz;
      ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
      ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

    // modify the coupling penalty parameter based on damage
    /*if(fd.damage < par->thresholdDamageForPenalty){
      fd.penaltyParameter = (1.0 - fd.damage) * fd.referencePenaltyParameterInternal;
    }
    else{
      PetscPrintf(PETSC_COMM_WORLD, "Damage Cap hit, not using damage!\n");
      exit(0);
      fd.penaltyParameter = (1.0 - par->thresholdDamageForPenalty) * fd.referencePenaltyParameterInternal;
    }*/

    for(i=0; i<dim; i++){
      velDiff[i] = fd.interpolatedVelocity[i] - fd.totalPhysicalVelocity[i];
    }

    for (a=0;a<user->nen;a++){
      PetscInt GlobalID = prb->map[a];

      for(i=0; i<dim; i++){

        value = fd.penaltyParameter
         * (prb->shape[0][a]
         * (velDiff[0]*fd.normal[0] + velDiff[1]*fd.normal[1] + velDiff[2]*fd.normal[2])*fd.normal[i]) //(v dot n)n
         * fd.nodalVolume;

        ierr = VecSetValueLocal(vecRes,GlobalID*user->iga->dof+i+1,value,ADD_VALUES);CHKERRQ(ierr);

      }
    }
  }
  put(fd_property, v, fd);
  put(info_property, v, info);
  }

  ierr = VecAssemblyBegin(vecRes);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecRes);CHKERRQ(ierr);
  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "getInterpolatedVelocity"
PetscErrorCode getInterpolatedVelocity(PARAMETERS *par,
                                               AppCtx *user, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  PetscReal uf[5] = {0.0};
  PetscReal pt[3] = {0.0};
  IGAProbe prb;
  ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);
  IGAProbeSetCollective(prb, PETSC_FALSE);
  pair<OutEdgeIterator,OutEdgeIterator> its = out_edges(manager.myTaskVertex,manager.graph);

  for(auto it=its.first; it != its.second; ++it){
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);
    if(!info.isTask && fd.Inside==1 && fd.material==0){
    pt[0] =  info.currentCoord[0]/user->Lx;
    pt[1] =  info.currentCoord[1]/user->Ly;
    pt[2] =  info.currentCoord[2]/user->Lz;
    ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
    ierr = IGAProbeFormValue(prb,&uf[0]);CHKERRQ(ierr);

    fd.interpolatedVelocity[0] = uf[1];
    fd.interpolatedVelocity[1] = uf[2];
    fd.interpolatedVelocity[2] = uf[3];

    if(uf[0]<=0.0){
      PetscPrintf(PETSC_COMM_SELF,"Density < 0, Current Density error: getInterpolatedVelocity \n");
      exit(0);
    }
    }
    put(fd_property,v,fd);
    put(info_property,v,info);
  }
  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

////////////////////////////////////////////////////////////


//########## Main Function #############//
//TODO:
// Add parallel restart functionality for larger problems -- is now Necessary for Shells -- Restart Proc. limiting
// clean up main using new IGA-Flow-PD code structure

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[]) {
  PetscFunctionBegin;
  PetscErrorCode ierr;
  AppCtx user;
  PetscInt i, j;
  PetscInt num_proc;
  PetscMPIInt rank;
  int mpi_id = 0;
  int mpi_size = 1;
  char version[128];

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  Teuchos::RCP<Epetra_Comm> epetraComm;

  #ifdef HAVE_MPI
    MPI_Comm_size(PETSC_COMM_WORLD, &num_proc);
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_rank(PETSC_COMM_WORLD, &mpi_id);
    MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);
    epetraComm = Teuchos::RCP<Epetra_Comm>(new Epetra_MpiComm(PETSC_COMM_WORLD));
  #else
    epetraComm = Teuchos::RCP<Epetra_Comm>(new Epetra_SerialComm);
  #endif

  // Banner //
  PetscGetVersion(version,sizeof(version));
  if(mpi_id == 0){
    cout << "\n--Peridigm-PD Shells + Fully-Coupled Level Set Compressible flow (Penalty Coupling Approach)--" << endl ;
    cout << "Petsc Version\n" << version<< endl ;
    cout << "Using Boost "
          << BOOST_VERSION / 100000     << "."
          << BOOST_VERSION / 100 % 1000 << "."
          << BOOST_VERSION % 100
          << endl;
    if(mpi_size > 1){
      cout << "MPI initialized on " << mpi_size << " processors.\n" << endl;}
  }

  PARAMETERS *par;
  ierr = PetscMalloc1(sizeof(*par),&par);CHKERRQ(ierr);

  //Material properties of Water at shallow depth
  user.mu       = 8.9e-4;
  user.lamda    = -2.0*user.mu/3.0;
  user.kappa    = 0.6;
  user.Cv       = 4180.0;
  user.Cp       = 4181.1;
  user.p0       = 100000.0;
  user.max_its  = 3;
  user.thickness = 0.00126; //Thickness of plate
  user.debug = PETSC_FALSE;
  user.numLS_Fields = 1;

  /* Set discretization options */
  PetscInt p=2, C=PETSC_DECIDE;

  // MPI initialization
  boost::mpi::environment env(argc,argv);

  // create an empty particle manager
  //ParticleManager manager = ParticleManager();

  // seed random number generator based on MPI rank
  //srand(manager.myRank);

  par->initialTime  	   = 0.0;
  par->finalTime    	   = 0.5e-3;
  par->timeStep	         = 0.1e-7;
  par->currentTime  	   = par->initialTime;
  par->gamma			       = 0.5;
  par->stepNumber        = 0;
  par->FreqResults       = 25;
  par->densityRDX        = 1770.0;
  par->density           = 1420.0;//7.8e3; //Properties of composite plate
  par->youngModulus      = 78.24e9;//200e9;
  par->poissonRatio      = 0.0385;//0.3;

  // Parameters for Penalty based approach
  par->penaltyConstant = 9.0;
  par->DamageModeling = PETSC_FALSE;
  par->damageCriticalStress = 8.0e10;
  par->damageCriticalEpsilonPlastic = 0.2;
  par->thresholdDamageForPenalty = 0.9;

  user.TimeRestart       = 0;
  user.StepRestart       = 0;
  user.FreqRestarts      = 1;

  user.spacing  = 0.0000;
  user.Lx       = 0.15+user.spacing;
  user.Ly       = 0.15+user.spacing;
  user.Lz       = 0.15+user.spacing;
  user.horizon  = 3.0*0.305/59.0;

  user.PDInitialTime = par->initialTime;
  user.OutputRestart = par->finalTime;
  PetscBool set;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Blast-PD Options 3D","IGA");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-StepRestart","Step of the initial solution from file",__FILE__,user.StepRestart,&user.StepRestart,&set);CHKERRQ(ierr);
  if(set){
    user.PDInitialTime = user.StepRestart*par->timeStep;
    set = PETSC_FALSE;
  }
  ierr = PetscOptionsReal("-OutputRestart"," of restart cycles before PD should output its restart file",__FILE__,user.OutputRestart,&user.OutputRestart,&set);CHKERRQ(ierr);
  if(set){
    user.OutputRestart = user.StepRestart*par->timeStep + user.OutputRestart*par->timeStep*user.FreqRestarts;
    set = PETSC_FALSE;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (C == PETSC_DECIDE) C = p-1;


  int RestartStep = round(user.OutputRestart/par->timeStep);

  //IGA Setup
  PetscInt dim = 3;
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,6);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGARead(iga,"./Geo/Geometry.dat");CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = IGAWrite(iga,"Output/igaF.dat");CHKERRQ(ierr);
  user.iga = iga;

  //Create solution vector V (velocities) and A (accelerations) and dispacements D
  PetscReal t=0;
  ierr = IGACreateVec(iga,&user.V0);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&user.A0);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&user.D0);CHKERRQ(ierr);
  ierr = IGACreateVec(iga,&user.dA);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.V0);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.A0);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.D0);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.dA);CHKERRQ(ierr);

  ierr = PetscMalloc1(iga->proc_sizes[0],&user.processor_numElX);CHKERRQ(ierr);
  ierr = PetscMalloc1(iga->proc_sizes[1],&user.processor_numElY);CHKERRQ(ierr);
  ierr = PetscMalloc1(iga->proc_sizes[2],&user.processor_numElZ);CHKERRQ(ierr);

  PetscInt tempX[iga->proc_sizes[0]];
  PetscInt tempY[iga->proc_sizes[1]];
  PetscInt tempZ[iga->proc_sizes[2]];

    for (j=0;j<iga->proc_sizes[0];j++){
      user.processor_numElX[j] = 0;
      tempX[j] = 0;
    }
    for (j=0;j<iga->proc_sizes[1];j++){
      user.processor_numElY[j] = 0;
      tempY[j] = 0;
    }
    for (j=0;j<iga->proc_sizes[2];j++){
      user.processor_numElZ[j] = 0;
      tempZ[j] = 0;
    }

    tempX[iga->proc_ranks[0]] = iga->elem_width[0];
    tempY[iga->proc_ranks[1]] = iga->elem_width[1];
    tempZ[iga->proc_ranks[2]] = iga->elem_width[2];

  // Debugging
  // PetscPrintf(PETSC_COMM_WORLD, "proc_sizes = %d %d %d \n", iga->proc_sizes[0], iga->proc_sizes[1], iga->proc_sizes[2]);
  // PetscPrintf(PETSC_COMM_WORLD, "proc_ranks = %d %d %d \n", iga->proc_ranks[0], iga->proc_ranks[1], iga->proc_ranks[2]);
  // PetscPrintf(PETSC_COMM_WORLD, "elem_width = %d %d %d \n", iga->elem_width[0], iga->elem_width[1], iga->elem_width[2]);

  ierr = MPI_Allreduce(&tempX,user.processor_numElX,iga->proc_sizes[0],MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&tempY,user.processor_numElY,iga->proc_sizes[1],MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&tempZ,user.processor_numElZ,iga->proc_sizes[2],MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);

  for (j=0;j<iga->proc_sizes[0];j++) {
  //  PetscPrintf(PETSC_COMM_WORLD, "Proc # X = %d\n", user.processor_numElX[j]);
    user.processor_numElX[j] /= iga->proc_sizes[1]*iga->proc_sizes[2];}
  for (j=0;j<iga->proc_sizes[1];j++){
  //  PetscPrintf(PETSC_COMM_WORLD, "Proc # Y = %d\n", user.processor_numElY[j]);
    user.processor_numElY[j] /= iga->proc_sizes[0]*iga->proc_sizes[2];}
  for (j=0;j<iga->proc_sizes[2];j++){
  //  PetscPrintf(PETSC_COMM_WORLD, "Proc # Z = %d\n", user.processor_numElZ[j]);
    user.processor_numElZ[j] /= iga->proc_sizes[0]*iga->proc_sizes[1];}

  Vec Res;
  ierr = IGACreateVec(iga,&Res);CHKERRQ(ierr);
  ierr = VecZeroEntries(Res);CHKERRQ(ierr);

  Vec Density;
  ierr = IGACreateVec(iga,&Density);CHKERRQ(ierr);
  ierr = VecZeroEntries(Density);CHKERRQ(ierr);

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  for(PetscInt dir=0;dir<dim;dir++){
    for(PetscInt side=0;side<2;side++){
      ierr = IGASetBoundaryForm(iga,dir,side,PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  ierr =TSSetRadius_GeneralizedAlpha(&user,0.5);CHKERRQ(ierr);

  ierr = VecDuplicate(user.A0,&user.A1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.Ap);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.Aa);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.V1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.Vp);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.Va);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.D1);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.Dp);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0,&user.Da);CHKERRQ(ierr);

  //Initialize a new IGA with only LS DOF so Eikonal equation can be
  //iterated on easily without having to make new structures with different
  //degrees of freedom than the main IGA.
  IGA iga_LS;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga_LS);CHKERRQ(ierr);
  ierr = IGASetDim(iga_LS,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga_LS,1);CHKERRQ(ierr);
  ierr = IGAGetDim(iga_LS,&dim);CHKERRQ(ierr);
  ierr = IGARead(iga_LS,"./Geo/Geometry.dat");CHKERRQ(ierr);
  ierr = IGASetUp(iga_LS);CHKERRQ(ierr);
  ierr = IGAWrite(iga_LS,"./Output/igaF_LS.dat");CHKERRQ(ierr);
  user.iga_LS = iga_LS;

  ierr = IGACreateVec(iga_LS,&user.V0_LS);CHKERRQ(ierr);
  ierr = IGACreateVec(iga_LS,&user.A0_LS);CHKERRQ(ierr);
  ierr = IGACreateVec(iga_LS,&user.D0_LS);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.V0_LS);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.A0_LS);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.D0_LS);CHKERRQ(ierr);

  ierr = IGACreateVec(iga_LS,&user.dA_LS);CHKERRQ(ierr);
  ierr = VecZeroEntries(user.dA_LS);CHKERRQ(ierr);

  Vec Res_LS;
  ierr = IGACreateVec(iga_LS,&Res_LS);CHKERRQ(ierr);
  ierr = VecZeroEntries(Res_LS);CHKERRQ(ierr);

  //ierr = IGASetFormIFunction(iga_LS,Residual_LS,&user);CHKERRQ(ierr);

  for(PetscInt dir=0;dir<dim;dir++){
    for(PetscInt side=0;side<2;side++){
      ierr = IGASetBoundaryForm(iga_LS,dir,side,PETSC_TRUE);CHKERRQ(ierr);
    }
  }

  ierr = VecDuplicate(user.A0_LS,&user.A1_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.Ap_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.Aa_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.V1_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.Vp_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.Va_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.D1_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.Dp_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.Da_LS);CHKERRQ(ierr);
  ierr = VecDuplicate(user.A0_LS,&user.PhiReference);CHKERRQ(ierr);

  IGA iga_LSC;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga_LSC);CHKERRQ(ierr);
  ierr = IGASetDim(iga_LSC,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga_LSC,1);CHKERRQ(ierr);
  ierr = IGARead(iga_LSC,"./Geo/Geometry.dat");CHKERRQ(ierr);
  ierr = IGASetUp(iga_LSC);CHKERRQ(ierr);
  user.iga_LSC = iga_LSC;


  par->stepNumber  = user.StepRestart;
  par->currentTime = user.StepRestart*par->timeStep;

  //Initial condition or Read from Restarts
  if (user.StepRestart == 0){
    ierr = FormInitialCondition(iga,t,user.V0,&user);CHKERRQ(ierr);
  }

  // Check the mass of explosive in the initial condition //
  ierr = IGASetFormIFunction(iga,Residual_massCalc,&user);CHKERRQ(ierr);
  ierr = IGAComputeIFunction(iga,0.0,user.A0,0.0,user.V0,Density);CHKERRQ(ierr);
  MPI_Barrier(PETSC_COMM_WORLD);
  double m1;
  VecNorm(Density,NORM_1,&m1);
  PetscPrintf(PETSC_COMM_WORLD,"Total Initial Explosive Mass on Domain = %e [mg]\n", m1*1.0e6);
  // // ///////////////////////////////////////////////////// //

  ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
  // else {
  //   //ierr = ReadLastResults(par,user.V0,user.A0,user.StepRestart,user.FreqRestarts,manager,&user);CHKERRQ(ierr);
  //   //manager.connectVertsToTasks(false,&user);
  //   //manager.sync();
  //   // MPI_Allreduce(&user.totalInitialExplosiveVolume, &totalInitialExplosiveVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //   // MPI_Allreduce(&par->numNodes, &totalNumNodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  //   // PetscPrintf(PETSC_COMM_WORLD, "Read %d Immersed Particles from %d Foreground files\n", totalNumNodes, mpi_size);
  // }
  // //manager.sync();

     // Dump Initial Solution
     char filename[256];
     sprintf(filename,"Output/velS%d.dat",par->stepNumber);
     ierr = IGAWriteVec(user.iga,user.V0,filename);CHKERRQ(ierr);

  user.numFluidNodes =  0;

  const PetscScalar *arrayU;
  const PetscScalar *arrayV;
  Vec               localV;
  Vec               localU;

  PetscReal Ynorm = 0.0;
  PetscReal Ynorm0 = 0.0;
  while(par->currentTime+par->timeStep <= par->finalTime+1e-9){
    PetscPrintf(PETSC_COMM_WORLD,"######################################################### \n");
    PetscPrintf(PETSC_COMM_WORLD,"Step Number: %d  Time step: %e, Time: %e \n",par->stepNumber, par->timeStep, par->currentTime);
    PetscPrintf(PETSC_COMM_WORLD,"######################################################### \n");

    //// Time Step Adaptivity : In explosive expansion problems, the early timestep is
    //// limited due to the high initial element-wise Peclet number. However, after sufficient
    //// expansion has taken place, the stable timestep >> initial stable timestep. This process
    //// will allow the timestep to grow as the initial high velocites on the domain die off.
    if(par->stepNumber > 300 && par->timeStep < 0.5e-6){
      if(Ynorm <= 0.5*Ynorm0){
        par->timeStep *= 1.01;
      }
      if(Ynorm > 0.5*Ynorm0){
        par->timeStep *= 0.97;
      }
    }
    //// As long as the stabilizing timestep correction is more aggressive
    // than the increase, the simulation is stable.
    ///////////////////////////////////////////////////////////////////////////////////////
    par->currentTime+=par->timeStep;
    // Record the Timestep for post-processing.
    outputTimeData(&user, par);
    user.stepNumber  = par->stepNumber;

    //Fluid predictor
    ierr = TSPredictStage_GeneralizedAlpha(&user,par);CHKERRQ(ierr);

    PetscInt i,j,k,m,l, it;
    PetscInt dof = iga->dof;
    PetscInt dim = iga->dim;

    Mat A0;
    ierr = IGACreateMat(iga,&A0);CHKERRQ(ierr);

    for (it=0; it<user.max_its ; it++) {
       PetscPrintf(PETSC_COMM_WORLD,":::::::::::::::::::::::::::::::::\n");
       PetscPrintf(PETSC_COMM_WORLD,"  Iteration: %d  \n", it);
       PetscPrintf(PETSC_COMM_WORLD,":::::::::::::::::::::::::::::::::\n");
       // Check the mass of explosive in the initial condition //
       ierr = IGASetFormIFunction(iga,Residual_massCalc,&user);CHKERRQ(ierr);
       ierr = IGAComputeIFunction(iga,0.0,user.A0,0.0,user.V0,Density);CHKERRQ(ierr);
       MPI_Barrier(PETSC_COMM_WORLD);
       double m1;
       VecNorm(Density,NORM_1,&m1);
       PetscPrintf(PETSC_COMM_WORLD,"Current Explosive Mass = %e [mg]\n", m1*1.0e6);
       // // ///////////////////////////////////////////////////// //

       ierr = IGASetFormIFunction(iga,Residual,&user);CHKERRQ(ierr);
       user.it = it;

       ierr = TSUpdateAlphaLevels_GeneralizedAlpha(&user);CHKERRQ(ierr);

       PetscReal t = par->currentTime;
       PetscReal dt = par->timeStep;
       PetscReal Alpha_m = user.Alpha_m;
       PetscReal Alpha_f = user.Alpha_f;
       par->Alpha_f = Alpha_f;
       PetscReal stage_time = t + Alpha_f*dt;

      ierr = IGAGetLocalVecArray(iga,user.Va,&localU,&arrayU);CHKERRQ(ierr);
      ierr = IGAGetLocalVecArray(iga,user.Aa,&localV,&arrayV);CHKERRQ(ierr);

      ///// Integrated Tangent ///////
      ierr = IGAComputeIJacobianComp(iga,Alpha_m,user.Aa,stage_time,user.Va,A0,&user);CHKERRQ(ierr);
      MPI_Barrier(PETSC_COMM_WORLD);
      ///////////////////////////////

      ///// Compute Fluid residual contributions F+0 ////
      ierr = IGAComputeIFunction(iga,dt,user.Aa,stage_time,user.Va,Res);CHKERRQ(ierr);
      MPI_Barrier(PETSC_COMM_WORLD);
      ///////////////////////////////////////////////////

      ///////////////////////////////////////////////////
      ierr = IGARestoreLocalVecArray(iga,user.Va,&localU,&arrayU);CHKERRQ(ierr);
      ierr = IGARestoreLocalVecArray(iga,user.Aa,&localV,&arrayV);CHKERRQ(ierr);
      //////////////////////////

      PetscScalar bc_rho[6]={0.0};
      bc_rho[0]=1.0;
      PetscScalar bc_ux[6]={0.0};
      bc_ux[1]=1.0;
      PetscScalar bc_uy[6]={0.0};
      bc_uy[2]=1.0;
      PetscScalar bc_uz[6]={0.0};
      bc_uz[3]=1.0;
      PetscScalar bc_temp[6]={0.0};
      bc_temp[4]=1.0;

      //// Apply Boundary Conditions through A0 ////
      PetscReal h_x = user.Lx/user.iga->elem_sizes[0];
      PetscReal XX,YY,ZZ;
      PetscReal offset = 0.0;
      PetscInt nodesX  = iga->geom_lwidth[0], nodesY  = iga->geom_lwidth[1], nodesZ  = iga->geom_lwidth[2];
      PetscInt gnodesX = iga->geom_gwidth[0], gnodesY = iga->geom_gwidth[1];
      for(m=0;m<nodesZ;m++) {
        for(l=0;l<nodesY;l++) {
          for(k=0;k<nodesX;k++) {

             XX = iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim]+offset;
             YY = iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+1]+offset;
             ZZ = iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+2];

           //Reflective boundaries BC
           if((iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim] <= 0.00001+user.spacing/2.0) || (iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim] >= user.Lx-0.00001-user.spacing/2.0)) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             PetscInt index_array[6]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             index_array[5]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+5;
             MatSetValuesLocal(A0,1,&index,6,index_array,bc_ux,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,6,index_array,1,&index,bc_ux,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }

           if((iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+1] <=0.00001+user.spacing/2.0) || (iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+1] >= user.Ly-0.00001-user.spacing/2.0)) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             PetscInt index_array[6]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             index_array[5]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+5;
             MatSetValuesLocal(A0,1,&index,6,index_array,bc_uy,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,6,index_array,1,&index,bc_uy,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }

           if((iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+2] <=0.00001+user.spacing/2.0) || (iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+2] >= user.Lz-0.00001-user.spacing/2.0)) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             PetscInt index_array[6]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             index_array[5]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+5;
             MatSetValuesLocal(A0,1,&index,6,index_array,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,6,index_array,1,&index,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }
           // End boundary reflection

           //Isothermal condition (Necessary for water-JWL):
            if(PETSC_TRUE) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             PetscInt index_array[6]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             index_array[5]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+5;
             MatSetValuesLocal(A0,1,&index,6,index_array,bc_temp,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,6,index_array,1,&index,bc_temp,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }
        }
      }
    }
      //// End Boundary conditions ////////////////////

  /////////////////////////////
  ierr = MatAssemblyBegin(A0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);
  MPI_Barrier(PETSC_COMM_WORLD);
  /////////////////////////////

  ///// Check Residul ////////
  double ResNorm;
  VecNorm(Res,NORM_2,&ResNorm);
  PetscPrintf(PETSC_COMM_WORLD,"Res. Norm = %e\n", ResNorm);
  if (ResNorm != ResNorm) {
    PetscPrintf(PETSC_COMM_WORLD, "Residual = NaN!\n");
    exit(0);
  }
  ////////////////////////////

  // Obtain reference and current residual norm
  // for timestep conditioning. Conditioning on
  // the residual norm of the nonlinear iterations.
  if(it==0){Ynorm0 = ResNorm;}
  if(it==2){Ynorm  = ResNorm;}
  ////////////////////////////

  ///// Solve to find increment ////////////
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A0,A0);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(ksp, PETSC_TRUE);
  ierr = KSPSolve(ksp,Res,user.dA);CHKERRQ(ierr);
  ierr = TSUpdateStage_GeneralizedAlpha(&user,par,user.dA);CHKERRQ(ierr);
  MPI_Barrier(PETSC_COMM_WORLD);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ////////////////////////////

}//###### END OF ITERATION LOOP ##########
ierr = VecCopy(user.V1,user.V0);CHKERRQ(ierr);
ierr = VecCopy(user.A1,user.A0);CHKERRQ(ierr);
ierr = VecCopy(user.D1,user.D0);CHKERRQ(ierr);


//##################################################################################################
//##################################################################################################

///////////////////////////////// Re-Distancing            ///////////////////////////////////////

//##################################################################################################
//##################################################################################################

ierr = VecZeroEntries(user.V0_LS);CHKERRQ(ierr);
ierr = VecZeroEntries(user.A0_LS);CHKERRQ(ierr);
ierr = VecZeroEntries(user.D0_LS);CHKERRQ(ierr);
ierr = VecZeroEntries(user.A1_LS);CHKERRQ(ierr);
ierr = VecZeroEntries(user.D1_LS);CHKERRQ(ierr);
ierr = VecZeroEntries(user.V1_LS);CHKERRQ(ierr);
ierr = InitializeLS(iga,iga_LS,t,user.V1,user.V0_LS,&user,par);CHKERRQ(ierr);
ierr = VecCopy(user.V0_LS, user.PhiReference);

PetscReal t_LS = 0.0;
par->timeTildeFactor = 0.002;
PetscReal dt_LS = par->timeStep/par->timeTildeFactor;

double ResNormLS = 0.0;

PetscReal Alpha_f = user.Alpha_f;
PetscReal Alpha_m = user.Alpha_m;

 for(int II = 0 ; II < 5 ; II++){
   PetscPrintf(PETSC_COMM_WORLD,"----------------------------------------\n");
   PetscPrintf(PETSC_COMM_WORLD,"  Re-distance Step: %d \n", II);
   PetscPrintf(PETSC_COMM_WORLD,"----------------------------------------\n");

   Mat A0_LS;
   ierr = IGACreateMat(iga_LS,&A0_LS);CHKERRQ(ierr);

   // Predictor
   ierr = TSPredictStage_GeneralizedAlpha_LS(&user,par);CHKERRQ(ierr);

   for(int RD_IT = 0 ; RD_IT < 3 ; RD_IT++){

    PetscPrintf(PETSC_COMM_WORLD,"    RD iteration: %d \n",RD_IT);
    PetscReal stage_time_LS = t_LS + Alpha_f*dt_LS;
    //Alpha level Fluid
    //ierr = TSUpdateAlphaLevels_GeneralizedAlpha_LS(&user);CHKERRQ(ierr);
    ierr = VecWAXPY(user.Va_LS,-1.0,user.V0_LS,user.V1_LS);CHKERRQ(ierr);
    ierr = VecAYPX (user.Va_LS,user.Alpha_f,user.V0_LS);CHKERRQ(ierr);

    ierr = VecWAXPY(user.Aa_LS,-1.0,user.A0_LS,user.A1_LS);CHKERRQ(ierr);
    ierr = VecAYPX (user.Aa_LS,user.Alpha_m,user.A0_LS);CHKERRQ(ierr);

    ierr = VecWAXPY(user.Da_LS,-1.0,user.D0_LS,user.D1_LS);CHKERRQ(ierr);
    ierr = VecAYPX (user.Da_LS,user.Alpha_f,user.D0_LS);CHKERRQ(ierr);

    ierr = MatZeroEntries(A0_LS);CHKERRQ(ierr);
    ierr = IGAComputeIJacobianComp_LS(iga_LS,Alpha_m,user.Aa_LS,stage_time_LS,user.Va_LS, iga, user.Aa, user.Va, A0_LS, &user, par);CHKERRQ(ierr);
    ierr = VecZeroEntries(Res_LS);CHKERRQ(ierr);
    ierr = IGAComputeIFunction_LS(iga_LS, dt_LS, user.Aa_LS, stage_time_LS, user.Va_LS, iga, user.Aa, user.Va, Res_LS, par, &user);CHKERRQ(ierr);

    //Assembling the Tangent Matrix
    ierr = MatAssemblyBegin(A0_LS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A0_LS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   // //Assembling the Residual Vector
    ierr = VecAssemblyBegin(Res_LS);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res_LS);CHKERRQ(ierr);

    //Solving the system to get the acceleration increment on the background grid, user.dA_LS
    KSP ksp_LS;
    ierr = VecZeroEntries(user.dA_LS);CHKERRQ(ierr);
    ierr = IGACreateKSP(iga_LS,&ksp_LS);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp_LS,A0_LS,A0_LS);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(ksp_LS, PETSC_TRUE);
    ierr = KSPSetFromOptions(ksp_LS);CHKERRQ(ierr);
    ierr = KSPSolve(ksp_LS,Res_LS,user.dA_LS);CHKERRQ(ierr);

     //With dA we update the background quantities on the IGA_LS
     ierr = VecAXPY (user.A1_LS,-1.0,user.dA_LS);CHKERRQ(ierr);
     ierr = VecAXPY (user.V1_LS,-user.Gamma*dt_LS,user.dA_LS);CHKERRQ(ierr);
     ierr = VecAXPY (user.D1_LS,-user.Beta*dt_LS*dt_LS,user.dA_LS);CHKERRQ(ierr);
     ierr = KSPDestroy(&ksp_LS);CHKERRQ(ierr);

    ///// Check Residul ////////
    double incnorm;
    VecNorm(Res_LS,NORM_2,&ResNormLS);
    VecNorm(user.dA_LS,NORM_2,&incnorm);
    PetscPrintf(PETSC_COMM_WORLD,"      Res. Norm Level Set Field = %e | Inc. Norm Level Set Field = %e\n", ResNormLS, incnorm);

    if (ResNormLS != ResNormLS) {
      PetscPrintf(PETSC_COMM_WORLD, "Level Set Residual = NaN!\n");
      exit(0);
    }
    //////////////////////////

  }

  ierr = VecCopy(user.V1_LS,user.V0_LS);CHKERRQ(ierr);
  ierr = VecCopy(user.A1_LS,user.A0_LS);CHKERRQ(ierr);
  ierr = VecCopy(user.D1_LS,user.D0_LS);CHKERRQ(ierr);
  ierr = MatDestroy(&A0_LS);CHKERRQ(ierr);

 }

//##################################################################################################
//##################################################################################################

///////////////////////////////// Mass Correction Procedure ///////////////////////////////////////

//##################################################################################################
//##################################################################################################


 // We now have the Re-distanced phi, but it does not obey species mass
 //conservation yet. Introduce a correction procedure:

 ierr = IGACreateVec(iga_LSC,&user.V0_LSC);CHKERRQ(ierr);
 ierr = IGACreateVec(iga_LSC,&user.A0_LSC);CHKERRQ(ierr);
 ierr = IGACreateVec(iga_LSC,&user.dA_LSC);CHKERRQ(ierr);

 Vec Res_LSC;
 ierr = IGACreateVec(iga_LSC,&Res_LSC);CHKERRQ(ierr);
 ierr = VecZeroEntries(Res_LSC);CHKERRQ(ierr);

 for(PetscInt dir=0;dir<dim;dir++){
   for(PetscInt side=0;side<2;side++){
     ierr = IGASetBoundaryForm(iga_LSC,dir,side,PETSC_TRUE);CHKERRQ(ierr);
   }
 }

 Mat A0_LSC;
 ierr = IGACreateMat(iga_LSC,&A0_LSC);CHKERRQ(ierr);
 ierr = InitializeLSC(iga_LSC, user.V0_LSC);CHKERRQ(ierr);

  for(int massBalance = 0 ; massBalance < 5 ; massBalance++){

   ierr = MatZeroEntries(A0_LSC);CHKERRQ(ierr);
   ierr = IGAComputeIJacobianComp_LSC(iga_LSC, user.A0_LSC, user.V0_LSC, iga, user.A1, user.V1, iga_LS , user.PhiReference, user.V0_LS, A0_LSC , &user, par);CHKERRQ(ierr);
   ierr = VecZeroEntries(Res_LSC);CHKERRQ(ierr);
   ierr = IGAComputeIFunction_LSC    (iga_LSC, user.A0_LSC, user.V0_LSC, iga, user.A1, user.V1, iga_LS , user.PhiReference, user.V0_LS, Res_LSC, par  , &user);CHKERRQ(ierr);

   // Apply Dirchilet BC to SDF prior to mass correction
   PetscInt dof_LSC = iga_LSC->dof;

   ierr = MatAssemblyBegin(A0_LSC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A0_LSC,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = VecAssemblyBegin(Res_LSC);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(Res_LSC);CHKERRQ(ierr);
    double correctionNorm;
    VecNorm(Res_LSC,NORM_2,&correctionNorm);

    PetscPrintf(PETSC_COMM_WORLD, "                Mass Correction Res. Norm = %e               \n", correctionNorm);

    KSP ksp_LSC;
    ierr = VecZeroEntries(user.dA_LSC);CHKERRQ(ierr);
    ierr = IGACreateKSP(iga_LSC,&ksp_LSC);CHKERRQ(ierr);
    KSPSetErrorIfNotConverged(ksp_LSC, PETSC_TRUE);
    ierr = KSPSetOperators(ksp_LSC,A0_LSC,A0_LSC);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp_LSC);CHKERRQ(ierr);
    ierr = KSPSolve(ksp_LSC,Res_LSC,user.dA_LSC);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp_LSC);CHKERRQ(ierr);

    double incNorm;
    VecNorm(user.dA_LSC,NORM_2,&incNorm);
    PetscPrintf(PETSC_COMM_WORLD, "                 incNorm  = %e                               \n" , incNorm);

    ierr = VecAXPY(user.V0_LSC, -1.0, user.dA_LSC);CHKERRQ(ierr);
  }
  MPI_Barrier(PETSC_COMM_WORLD);


 // At this point, we have converged to the correction term, phi_prime. Now,
 // add this correction to the re-distanced phi to recover RD phi which satisfies
 // SMFC equation and is sharpened. Send the re-distanced SDF + correction back
 //to the main IGA and continue ; in this debugging version we will check the mass before and after
 //the correction term is applied.
 ierr = UpdatePhi(iga,iga_LS,iga_LSC,user.V0,user.V1_LS,user.V0_LSC,&user,par);CHKERRQ(ierr);
 MPI_Barrier(PETSC_COMM_WORLD);
// Clean up correction procedure and continue
 ierr = MatDestroy(&A0_LSC);CHKERRQ(ierr);

//////////////////////////////////////////////////////////////////////////////////////////////////////

  // Output any probe data that is required (Shouldnt be in different functions)
  ierr = probePressureSignals76(&user, par);
  ierr = probePressureSignals114(&user, par);
  ierr = probePressureSignals152(&user, par);

  //Update Solution State in PD and IGA
  par->stepNumber++;

   // Write Solution
   if (par->stepNumber % par->FreqResults == 0) {
        char filename[256];
        sprintf(filename,"./Output/velS%d.dat",par->stepNumber);
        ierr = IGAWriteVec(user.iga,user.V1,filename);CHKERRQ(ierr);
    }

    // Write Restarts & exit.
    if (par->stepNumber % (RestartStep) == 0) {

        MPI_Barrier(PETSC_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        char filename[256];
        sprintf(filename,"ResS%d.dat",par->stepNumber);
        ierr = IGAWriteVec(user.iga,Res,filename);CHKERRQ(ierr);
        sprintf(filename,"velS%d.dat",par->stepNumber);
        ierr = IGAWriteVec(user.iga,user.V1,filename);CHKERRQ(ierr);
        sprintf(filename,"acelS%d.dat",par->stepNumber);
        ierr = IGAWriteVec(user.iga,user.A1,filename);CHKERRQ(ierr);
        //ierr = OutputRestarts(par,user.V1,user.A1,manager);CHKERRQ(ierr);
        MPI_Barrier(PETSC_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        ierr = VecDestroy(&user.V0);CHKERRQ(ierr);
        ierr = VecDestroy(&user.Va);CHKERRQ(ierr);
        ierr = VecDestroy(&user.V1);CHKERRQ(ierr);

        ierr = VecDestroy(&user.A0);CHKERRQ(ierr);
        ierr = VecDestroy(&user.Aa);CHKERRQ(ierr);
        ierr = VecDestroy(&user.A1);CHKERRQ(ierr);

        ierr = VecDestroy(&user.D0);CHKERRQ(ierr);
        ierr = VecDestroy(&user.Da);CHKERRQ(ierr);
        ierr = VecDestroy(&user.D1);CHKERRQ(ierr);

        ierr = VecDestroy(&user.Vp);CHKERRQ(ierr);
        ierr = VecDestroy(&user.Ap);CHKERRQ(ierr);
        ierr = VecDestroy(&user.Dp);CHKERRQ(ierr);

        ierr = VecDestroy(&user.dA);CHKERRQ(ierr);
        PetscPrintf(PETSC_COMM_WORLD, "Restarts Written at time %e step %d. Exiting...\n", par->currentTime, par->stepNumber);
        MPI_Barrier(PETSC_COMM_WORLD);
        #ifdef HAVE_MPI
          PetscFree(par);CHKERRQ(ierr);
          IGADestroy(&iga);CHKERRQ(ierr);
          PetscFinalize();CHKERRQ(ierr);
        #endif
        PetscFunctionReturn(0);
    }

  ierr = MatDestroy(&A0);CHKERRQ(ierr);
  MPI_Barrier(PETSC_COMM_WORLD);
  // ierr = MatDestroy(&A0_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&Res_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.V0_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.Va_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.V1_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.A0_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.Aa_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.A1_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.D0_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.Da_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.D1_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.Vp_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.Ap_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.Dp_LS);CHKERRQ(ierr);
  // ierr = VecDestroy(&user.dA_LS);CHKERRQ(ierr);
  // ierr = IGADestroy(&iga_LS);CHKERRQ(ierr);

  }//End of Time Integration Loop

  ierr = VecDestroy(&user.V0);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Va);CHKERRQ(ierr);
  ierr = VecDestroy(&user.V1);CHKERRQ(ierr);

  ierr = VecDestroy(&user.A0);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Aa);CHKERRQ(ierr);
  ierr = VecDestroy(&user.A1);CHKERRQ(ierr);

  ierr = VecDestroy(&user.D0);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Da);CHKERRQ(ierr);
  ierr = VecDestroy(&user.D1);CHKERRQ(ierr);

  ierr = VecDestroy(&user.Vp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Ap);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Dp);CHKERRQ(ierr);

  ierr = VecDestroy(&user.dA);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Complete\n");

  #ifdef HAVE_MPI
  PetscFree(par);CHKERRQ(ierr);
  IGADestroy(&iga);CHKERRQ(ierr);
  PetscFinalize();CHKERRQ(ierr);
  #endif

  PetscFunctionReturn(0);
  }
