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

using namespace std;
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
  IGA iga_energy;
  IGA iga_strain;
  PetscReal Lx,Ly,Lz,mu,lamda,kappa,temp0,p0,R,Cp,Cv,spacing;
  Vec A1,Ap,An,V1,Vp,Vn,D1,Dp,Dn,Aa,Va,Da,V0,A0,D0,dA;
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
  PetscReal H[3];
  PetscReal TimeRestart;
  PetscInt  numFluidNodes;

  PetscInt  StepRestart;
  PetscInt  stepNumber;
  PetscInt  it;
  PetscInt  FreqRestarts;
  PetscInt xdivint;
  PetscInt ydivint;
  PetscInt zdivint;
  PetscInt nen;
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


  // obtain, in O(1) time, the rank of the processor whose subdomain contains
  // the point
  int pointRank(double *x, AppCtx *user){

    PetscInt elemX,elemY,elemZ;
    int i;

    //This has to be redone for non-uniform meshes:
    elemX = (int) (x[0]/user->H[0]);
    elemY = (int) (x[1]/user->H[1]);
    elemZ = (int) (x[2]/user->H[2]);
    if (elemX == user->iga->elem_sizes[0] && x[0] <= user->Lx){
      elemX = user->iga->elem_sizes[0]-1;
    }
    if(elemY == user->iga->elem_sizes[1] && x[1] <= user->Ly){
      elemY = user->iga->elem_sizes[1]-1;
    }
    if(elemZ == user->iga->elem_sizes[2] && x[2] <= user->Lz){
      elemZ = user->iga->elem_sizes[2]-1;
    }


    PetscInt rankX=-1,rankY=-1,rankZ=-1;
    PetscInt check = 0;
    for (i=0;i<user->iga->proc_sizes[0];i++){
      check += user->processor_numElX[i];
      if (elemX<check){
        rankX = i;
        break;
      }
    }
    check = 0;
    for (i=0;i<user->iga->proc_sizes[1];i++){
      check += user->processor_numElY[i];
      if (elemY<check){
        rankY = i;
        break;
      }
    }
    check = 0;
    for (i=0;i<user->iga->proc_sizes[2];i++){
      check += user->processor_numElZ[i];
      if (elemZ<check){
        rankZ = i;
        break;
      }
    }


    PetscInt myRank = rankZ*user->iga->proc_sizes[0]*user->iga->proc_sizes[1] + rankY*user->iga->proc_sizes[0]+rankX;
    if ((elemX < 0) || (elemY < 0) || (elemZ < 0)){
      myRank = -1;
      PetscPrintf(PETSC_COMM_SELF, "Error! Point at x = %e %e %e returned negative Element values: elem = %d %d %d with H = %e %e %e\n", x[0], x[1], x[2], elemX, elemY, elemZ, user->H[0], user->H[1], user->H[2]);
      exit(0);
    }
    if ((elemX >= user->iga->elem_sizes[0]) || (elemY >= user->iga->elem_sizes[1]) || (elemZ >= user->iga->elem_sizes[2])){
      PetscPrintf(PETSC_COMM_SELF, "Error! Point at x = %e %e %e returned Element values which exceed the number of elements: elem = %d %d %d with H = %e %e %e\n", x[0], x[1], x[2], elemX, elemY, elemZ, user->H[0], user->H[1], user->H[2]);
      myRank = -1;
      exit(0);
    }
    return myRank;
  } // end pointRank

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


} PARAMETERS;

typedef struct {
  PetscScalar rho,ux,uy,uz,temp;
} Field;
//// End Data structures for immersed-IGA ////

//// Re-definition of PetIGA functions ////
#undef  __FUNCT__
#define __FUNCT__ "IGALocateElement_1"
PetscBool IGALocateElement_1(IGA iga,PetscReal *pnt,IGAElement element)
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
    ID[i] = 0;
    PetscReal h = U[iga->axis[i]->p+1] - U[iga->axis[i]->p];
    PetscReal deltau = pnt[i] - U[0];
    e = (PetscInt) (deltau/h);
    ID[i] = e;
    /* find which nonzero span this point is located in */
   // for(j=0;j<m;j++){
   //    if(U[j+1]-U[j]>1.0e-13) e += 1;
   //    if(pnt[i] > U[j] && pnt[i] <= U[j+1]) ID[i] = e;
   //  }
    /* reject if the element is not in this partition */
     if(ID[i] < iga->elem_start[i] || ID[i] >= iga->elem_start[i]+iga->elem_width[i]) return PETSC_FALSE;
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

				Geo.write( (char*)&info.currentCoord[0], sizeof(double));
				Geo.write( (char*)&info.currentCoord[1], sizeof(double));
				Geo.write( (char*)&info.currentCoord[2], sizeof(double));

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
        Vol.write( (char*)& fd.referencePenaltyParameterInternal, sizeof(double));

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
#define __FUNCT__ "OutputOldGeometry"
PetscErrorCode OutputOldGeometry(PARAMETERS *par,ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscErrorCode ierr;
  PetscInt i,j;
  PetscInt count=par->numNodes;

  PetscInt num_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ostringstream convert1;


  string fname;


  PetscInt counter = 0; //Number of nodes stored in current rank
  BGL_FORALL_VERTICES(v,manager.graph,Graph){
 	 ParticleInfo info = get(info_property,v);
 	 if(!info.isTask){
 		 counter++;
 	 }
  }

     // ##################################################
     //                 Old Geometry File
     // ##################################################

  if (counter > 0){
 	  convert1 << "RestartGeo." << rank << "." << par->stepNumber << ".dat";
 	  fname = convert1.str();
 	  ofstream Geo;
 	  Geo.open(fname, ios::out | ios::binary);

        BGL_FORALL_VERTICES(v,manager.graph,Graph){
       	 ParticleInfo info = get(info_property,v);
       	 FieldData fd = get(fd_property,v);
       	if(!info.isTask){
       		Geo.write( (char*)&info.currentCoord[0], sizeof(double));
		Geo.write( (char*)&info.currentCoord[1], sizeof(double));
		Geo.write( (char*)&info.currentCoord[2], sizeof(double));

       	}

        }
        Geo.close();

  }

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

    // Geometry Old
    PetscInt temp1 = StepRestart-1;
    ostringstream convert3;
    convert3 << "RestartGeo." << rank << "." << temp1 << ".dat";
    fname = convert3.str();
    ifstream Geo1;
    Geo1.open(fname, ios::in | ios::binary);


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

    Geo.read( (char*)&info.currentCoord[0], sizeof(double));
	  Geo.read( (char*)&info.currentCoord[1], sizeof(double));
 	  Geo.read( (char*)&info.currentCoord[2], sizeof(double));

	  Geo1.read( (char*)&fd.totalPhysicalDisplacement[0], sizeof(double));
    Geo1.read( (char*)&fd.totalPhysicalDisplacement[1], sizeof(double));
    Geo1.read( (char*)&fd.totalPhysicalDisplacement[2], sizeof(double));

    fd.totalPhysicalDisplacement[0] = info.currentCoord[0] - fd.totalPhysicalDisplacement[0];
    fd.totalPhysicalDisplacement[1] = info.currentCoord[1] - fd.totalPhysicalDisplacement[1];
    fd.totalPhysicalDisplacement[2] = info.currentCoord[2] - fd.totalPhysicalDisplacement[2];

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
  Geo1.close();
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
  PetscErrorCode ierr;
  PetscInt i,j;

  ostringstream ElemInfo;
  ElemInfo << "ElementInfo" << ".dat";
  string ElemInfoName = ElemInfo.str();
  ifstream ELin;
  ELin.open(ElemInfoName.c_str());
  double xdiv, ydiv, zdiv;

  ELin >> xdiv;
  ELin >> ydiv;
  ELin >> zdiv;

  user->xdivint = (int)xdiv;
  user->ydivint = (int)ydiv;
  user->zdivint = (int)zdiv;

  PetscPrintf(PETSC_COMM_WORLD, "Loaded %d x %d x %d IGA Discretization\n\n", user->iga->axis[0]->nel, user->iga->axis[1]->nel, user->iga->axis[2]->nel);

  PetscMalloc1(user->xdivint, &user->xhat);
  PetscMalloc1(user->ydivint, &user->yhat);
  PetscMalloc1(user->zdivint, &user->zhat);

  for(int i = 0; i<user->xdivint; i++){
    ELin >> user->xhat[i];
    ELin >> user->yhat[i];
    ELin >> user->zhat[i];
  }
return 0;
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
  PetscReal hx = 0.0;
  //Uniform mesh assumed here
  PetscReal h = (user->Lx)/user->iga->elem_sizes[0];

  for(i=info.xs;i<info.xs+info.xm;i++){
    for(j=info.ys;j<info.ys+info.ym;j++){
      for(k=info.zs;k<info.zs+info.zm;k++){

        PetscReal x = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim];
        PetscReal y = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 1];
        PetscReal z = iga->geometryX[((k-info.zs)*gnodesX*gnodesY+(j-info.ys)*gnodesX+(i-info.xs))*dim + 2];

        PetscReal r = sqrt((x-user->Lx/2.0)*(x-user->Lx/2.0)+(y-user->Ly/2.0)*(y-user->Ly/2.0));

      //Shallow water Tank conditions
      u[k][j][i].ux   =  0.0;
      u[k][j][i].uz   =  0.0;
      u[k][j][i].uy   =  0.0;
      u[k][j][i].rho  =  1000.0;
      u[k][j][i].temp =  290.0;

      //RDX Charge
      PetscReal Standoff = 0.076;
      if(r<=(0.0049+1.5*h+0.000001) && sqrt((z-0.075)*(z-0.075))<=(0.00465132087901/2.0+1.5*h)/*z<=(30.0*h+(user->Lx)/user->iga->elem_sizes[0]) && z>=29.0*h-(user->Lx)/user->iga->elem_sizes[0]*/){
        u[k][j][i].ux   =  0.0;
        u[k][j][i].uz   =  0.0;
        u[k][j][i].uy   =  0.0;
        u[k][j][i].rho  =  1770.0;
        u[k][j][i].temp =  290.0;
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
#undef __FUNCT__
#define __FUNCT__ "Invert5x5"
PetscErrorCode Invert5x5( PetscScalar (*A)[5],PetscReal (*Ainv)[5],void *ctx)
{

    AppCtx *user = (AppCtx *)ctx;

    PetscReal DET;
    PetscInt  i,j;
    PetscInt  dof    = user->iga->dof;


    PetscReal A11, A12, A13, A14, A15;
    PetscReal A21, A22, A23, A24, A25;
    PetscReal A31, A32, A33, A34, A35;
    PetscReal A41, A42, A43, A44, A45;
    PetscReal A51, A52, A53, A54, A55;



    A11=A[0][0]; A12=A[0][1]; A13=A[0][2]; A14=A[0][3]; A15=A[0][4];
    A21=A[1][0]; A22=A[1][1]; A23=A[1][2]; A24=A[1][3]; A25=A[1][4];
    A31=A[2][0]; A32=A[2][1]; A33=A[2][2]; A34=A[2][3]; A35=A[2][4];
    A41=A[3][0]; A42=A[3][1]; A43=A[3][2]; A44=A[3][3]; A45=A[3][4];
    A51=A[4][0]; A52=A[4][1]; A53=A[4][2]; A54=A[4][3]; A55=A[4][4];

    DET = A15*A24*A33*A42*A51-A14*A25*A33*A42*A51-A15*A23*A34*A42*A51+
    A13*A25*A34*A42*A51+A14*A23*A35*A42*A51-A13*A24*A35*A42*A51-
    A15*A24*A32*A43*A51+A14*A25*A32*A43*A51+A15*A22*A34*A43*A51-
    A12*A25*A34*A43*A51-A14*A22*A35*A43*A51+A12*A24*A35*A43*A51+
    A15*A23*A32*A44*A51-A13*A25*A32*A44*A51-A15*A22*A33*A44*A51+
    A12*A25*A33*A44*A51+A13*A22*A35*A44*A51-A12*A23*A35*A44*A51-
    A14*A23*A32*A45*A51+A13*A24*A32*A45*A51+A14*A22*A33*A45*A51-
    A12*A24*A33*A45*A51-A13*A22*A34*A45*A51+A12*A23*A34*A45*A51-
    A15*A24*A33*A41*A52+A14*A25*A33*A41*A52+A15*A23*A34*A41*A52-
    A13*A25*A34*A41*A52-A14*A23*A35*A41*A52+A13*A24*A35*A41*A52+
    A15*A24*A31*A43*A52-A14*A25*A31*A43*A52-A15*A21*A34*A43*A52+
    A11*A25*A34*A43*A52+A14*A21*A35*A43*A52-A11*A24*A35*A43*A52-
    A15*A23*A31*A44*A52+A13*A25*A31*A44*A52+A15*A21*A33*A44*A52-
    A11*A25*A33*A44*A52-A13*A21*A35*A44*A52+A11*A23*A35*A44*A52+
    A14*A23*A31*A45*A52-A13*A24*A31*A45*A52-A14*A21*A33*A45*A52+
    A11*A24*A33*A45*A52+A13*A21*A34*A45*A52-A11*A23*A34*A45*A52+
    A15*A24*A32*A41*A53-A14*A25*A32*A41*A53-A15*A22*A34*A41*A53+
    A12*A25*A34*A41*A53+A14*A22*A35*A41*A53-A12*A24*A35*A41*A53-
    A15*A24*A31*A42*A53+A14*A25*A31*A42*A53+A15*A21*A34*A42*A53-
    A11*A25*A34*A42*A53-A14*A21*A35*A42*A53+A11*A24*A35*A42*A53+
    A15*A22*A31*A44*A53-A12*A25*A31*A44*A53-A15*A21*A32*A44*A53+
    A11*A25*A32*A44*A53+A12*A21*A35*A44*A53-A11*A22*A35*A44*A53-
    A14*A22*A31*A45*A53+A12*A24*A31*A45*A53+A14*A21*A32*A45*A53-
    A11*A24*A32*A45*A53-A12*A21*A34*A45*A53+A11*A22*A34*A45*A53-
    A15*A23*A32*A41*A54+A13*A25*A32*A41*A54+A15*A22*A33*A41*A54-
    A12*A25*A33*A41*A54-A13*A22*A35*A41*A54+A12*A23*A35*A41*A54+
    A15*A23*A31*A42*A54-A13*A25*A31*A42*A54-A15*A21*A33*A42*A54+
    A11*A25*A33*A42*A54+A13*A21*A35*A42*A54-A11*A23*A35*A42*A54-
    A15*A22*A31*A43*A54+A12*A25*A31*A43*A54+A15*A21*A32*A43*A54-
    A11*A25*A32*A43*A54-A12*A21*A35*A43*A54+A11*A22*A35*A43*A54+
    A13*A22*A31*A45*A54-A12*A23*A31*A45*A54-A13*A21*A32*A45*A54+
    A11*A23*A32*A45*A54+A12*A21*A33*A45*A54-A11*A22*A33*A45*A54+
    A14*A23*A32*A41*A55-A13*A24*A32*A41*A55-A14*A22*A33*A41*A55+
    A12*A24*A33*A41*A55+A13*A22*A34*A41*A55-A12*A23*A34*A41*A55-
    A14*A23*A31*A42*A55+A13*A24*A31*A42*A55+A14*A21*A33*A42*A55-
    A11*A24*A33*A42*A55-A13*A21*A34*A42*A55+A11*A23*A34*A42*A55+
    A14*A22*A31*A43*A55-A12*A24*A31*A43*A55-A14*A21*A32*A43*A55+
    A11*A24*A32*A43*A55+A12*A21*A34*A43*A55-A11*A22*A34*A43*A55-
    A13*A22*A31*A44*A55+A12*A23*A31*A44*A55+A13*A21*A32*A44*A55-
    A11*A23*A32*A44*A55-A12*A21*A33*A44*A55+A11*A22*A33*A44*A55;



    PetscReal COFACTOR[5][5]={{0.0}};


    COFACTOR[0][0] = A25*A34*A43*A52-A24*A35*A43*A52-A25*A33*A44*A52+
    A23*A35*A44*A52+A24*A33*A45*A52-A23*A34*A45*A52-A25*A34*A42*A53+
    A24*A35*A42*A53+A25*A32*A44*A53-A22*A35*A44*A53-A24*A32*A45*A53+
    A22*A34*A45*A53+A25*A33*A42*A54-A23*A35*A42*A54-A25*A32*A43*A54+
    A22*A35*A43*A54+A23*A32*A45*A54-A22*A33*A45*A54-A24*A33*A42*A55+
    A23*A34*A42*A55+A24*A32*A43*A55-A22*A34*A43*A55-A23*A32*A44*A55+
    A22*A33*A44*A55;

    COFACTOR[1][0] = -A15*A34*A43*A52+A14*A35*A43*A52+A15*A33*A44*A52-
    A13*A35*A44*A52-A14*A33*A45*A52+A13*A34*A45*A52+A15*A34*A42*A53-
    A14*A35*A42*A53-A15*A32*A44*A53+A12*A35*A44*A53+A14*A32*A45*A53-
    A12*A34*A45*A53-A15*A33*A42*A54+A13*A35*A42*A54+A15*A32*A43*A54-
    A12*A35*A43*A54-A13*A32*A45*A54+A12*A33*A45*A54+A14*A33*A42*A55-
    A13*A34*A42*A55-A14*A32*A43*A55+A12*A34*A43*A55+A13*A32*A44*A55-
    A12*A33*A44*A55;

    COFACTOR[2][0] = A15*A24*A43*A52-A14*A25*A43*A52-A15*A23*A44*A52+
    A13*A25*A44*A52+A14*A23*A45*A52-A13*A24*A45*A52-A15*A24*A42*A53+
    A14*A25*A42*A53+A15*A22*A44*A53-A12*A25*A44*A53-A14*A22*A45*A53+
    A12*A24*A45*A53+A15*A23*A42*A54-A13*A25*A42*A54-A15*A22*A43*A54+
    A12*A25*A43*A54+A13*A22*A45*A54-A12*A23*A45*A54-A14*A23*A42*A55+
    A13*A24*A42*A55+A14*A22*A43*A55-A12*A24*A43*A55-A13*A22*A44*A55+
    A12*A23*A44*A55;

    COFACTOR[3][0] = -A15*A24*A33*A52+A14*A25*A33*A52+A15*A23*A34*A52-
    A13*A25*A34*A52-A14*A23*A35*A52+A13*A24*A35*A52+A15*A24*A32*A53-
    A14*A25*A32*A53-A15*A22*A34*A53+A12*A25*A34*A53+A14*A22*A35*A53-
    A12*A24*A35*A53-A15*A23*A32*A54+A13*A25*A32*A54+A15*A22*A33*A54-
    A12*A25*A33*A54-A13*A22*A35*A54+A12*A23*A35*A54+A14*A23*A32*A55-
    A13*A24*A32*A55-A14*A22*A33*A55+A12*A24*A33*A55+A13*A22*A34*A55-
    A12*A23*A34*A55;

    COFACTOR[4][0] = A15*A24*A33*A42-A14*A25*A33*A42-A15*A23*A34*A42+
    A13*A25*A34*A42+A14*A23*A35*A42-A13*A24*A35*A42-A15*A24*A32*A43+
    A14*A25*A32*A43+A15*A22*A34*A43-A12*A25*A34*A43-A14*A22*A35*A43+
    A12*A24*A35*A43+A15*A23*A32*A44-A13*A25*A32*A44-A15*A22*A33*A44+
    A12*A25*A33*A44+A13*A22*A35*A44-A12*A23*A35*A44-A14*A23*A32*A45+
    A13*A24*A32*A45+A14*A22*A33*A45-A12*A24*A33*A45-A13*A22*A34*A45+
    A12*A23*A34*A45;

    COFACTOR[0][1] = -A25*A34*A43*A51+A24*A35*A43*A51+A25*A33*A44*A51-
    A23*A35*A44*A51-A24*A33*A45*A51+A23*A34*A45*A51+A25*A34*A41*A53-
    A24*A35*A41*A53-A25*A31*A44*A53+A21*A35*A44*A53+A24*A31*A45*A53-
    A21*A34*A45*A53-A25*A33*A41*A54+A23*A35*A41*A54+A25*A31*A43*A54-
    A21*A35*A43*A54-A23*A31*A45*A54+A21*A33*A45*A54+A24*A33*A41*A55-
    A23*A34*A41*A55-A24*A31*A43*A55+A21*A34*A43*A55+A23*A31*A44*A55-
    A21*A33*A44*A55;

    COFACTOR[1][1] = A15*A34*A43*A51-A14*A35*A43*A51-A15*A33*A44*A51+
    A13*A35*A44*A51+A14*A33*A45*A51-A13*A34*A45*A51-A15*A34*A41*A53+
    A14*A35*A41*A53+A15*A31*A44*A53-A11*A35*A44*A53-A14*A31*A45*A53+
    A11*A34*A45*A53+A15*A33*A41*A54-A13*A35*A41*A54-A15*A31*A43*A54+
    A11*A35*A43*A54+A13*A31*A45*A54-A11*A33*A45*A54-A14*A33*A41*A55+
    A13*A34*A41*A55+A14*A31*A43*A55-A11*A34*A43*A55-A13*A31*A44*A55+
    A11*A33*A44*A55;

    COFACTOR[2][1] = -A15*A24*A43*A51+A14*A25*A43*A51+A15*A23*A44*A51-
    A13*A25*A44*A51-A14*A23*A45*A51+A13*A24*A45*A51+A15*A24*A41*A53-
    A14*A25*A41*A53-A15*A21*A44*A53+A11*A25*A44*A53+A14*A21*A45*A53-
    A11*A24*A45*A53-A15*A23*A41*A54+A13*A25*A41*A54+A15*A21*A43*A54-
    A11*A25*A43*A54-A13*A21*A45*A54+A11*A23*A45*A54+A14*A23*A41*A55-
    A13*A24*A41*A55-A14*A21*A43*A55+A11*A24*A43*A55+A13*A21*A44*A55-
    A11*A23*A44*A55;

    COFACTOR[3][1] = A15*A24*A33*A51-A14*A25*A33*A51-A15*A23*A34*A51+
    A13*A25*A34*A51+A14*A23*A35*A51-A13*A24*A35*A51-A15*A24*A31*A53+
    A14*A25*A31*A53+A15*A21*A34*A53-A11*A25*A34*A53-A14*A21*A35*A53+
    A11*A24*A35*A53+A15*A23*A31*A54-A13*A25*A31*A54-A15*A21*A33*A54+
    A11*A25*A33*A54+A13*A21*A35*A54-A11*A23*A35*A54-A14*A23*A31*A55+
    A13*A24*A31*A55+A14*A21*A33*A55-A11*A24*A33*A55-A13*A21*A34*A55+
    A11*A23*A34*A55;

    COFACTOR[4][1] = -A15*A24*A33*A41+A14*A25*A33*A41+A15*A23*A34*A41-
    A13*A25*A34*A41-A14*A23*A35*A41+A13*A24*A35*A41+A15*A24*A31*A43-
    A14*A25*A31*A43-A15*A21*A34*A43+A11*A25*A34*A43+A14*A21*A35*A43-
    A11*A24*A35*A43-A15*A23*A31*A44+A13*A25*A31*A44+A15*A21*A33*A44-
    A11*A25*A33*A44-A13*A21*A35*A44+A11*A23*A35*A44+A14*A23*A31*A45-
    A13*A24*A31*A45-A14*A21*A33*A45+A11*A24*A33*A45+A13*A21*A34*A45-
    A11*A23*A34*A45;

    COFACTOR[0][2] = A25*A34*A42*A51-A24*A35*A42*A51-A25*A32*A44*A51+
    A22*A35*A44*A51+A24*A32*A45*A51-A22*A34*A45*A51-A25*A34*A41*A52+
    A24*A35*A41*A52+A25*A31*A44*A52-A21*A35*A44*A52-A24*A31*A45*A52+
    A21*A34*A45*A52+A25*A32*A41*A54-A22*A35*A41*A54-A25*A31*A42*A54+
    A21*A35*A42*A54+A22*A31*A45*A54-A21*A32*A45*A54-A24*A32*A41*A55+
    A22*A34*A41*A55+A24*A31*A42*A55-A21*A34*A42*A55-A22*A31*A44*A55+
    A21*A32*A44*A55;

    COFACTOR[1][2] = -A15*A34*A42*A51+A14*A35*A42*A51+A15*A32*A44*A51-
    A12*A35*A44*A51-A14*A32*A45*A51+A12*A34*A45*A51+A15*A34*A41*A52-
    A14*A35*A41*A52-A15*A31*A44*A52+A11*A35*A44*A52+A14*A31*A45*A52-
    A11*A34*A45*A52-A15*A32*A41*A54+A12*A35*A41*A54+A15*A31*A42*A54-
    A11*A35*A42*A54-A12*A31*A45*A54+A11*A32*A45*A54+A14*A32*A41*A55-
    A12*A34*A41*A55-A14*A31*A42*A55+A11*A34*A42*A55+A12*A31*A44*A55-
    A11*A32*A44*A55;

    COFACTOR[2][2] = A15*A24*A42*A51-A14*A25*A42*A51-A15*A22*A44*A51+
    A12*A25*A44*A51+A14*A22*A45*A51-A12*A24*A45*A51-A15*A24*A41*A52+
    A14*A25*A41*A52+A15*A21*A44*A52-A11*A25*A44*A52-A14*A21*A45*A52+
    A11*A24*A45*A52+A15*A22*A41*A54-A12*A25*A41*A54-A15*A21*A42*A54+
    A11*A25*A42*A54+A12*A21*A45*A54-A11*A22*A45*A54-A14*A22*A41*A55+
    A12*A24*A41*A55+A14*A21*A42*A55-A11*A24*A42*A55-A12*A21*A44*A55+
    A11*A22*A44*A55;

    COFACTOR[3][2] = -A15*A24*A32*A51+A14*A25*A32*A51+A15*A22*A34*A51-
    A12*A25*A34*A51-A14*A22*A35*A51+A12*A24*A35*A51+A15*A24*A31*A52-
    A14*A25*A31*A52-A15*A21*A34*A52+A11*A25*A34*A52+A14*A21*A35*A52-
    A11*A24*A35*A52-A15*A22*A31*A54+A12*A25*A31*A54+A15*A21*A32*A54-
    A11*A25*A32*A54-A12*A21*A35*A54+A11*A22*A35*A54+A14*A22*A31*A55-
    A12*A24*A31*A55-A14*A21*A32*A55+A11*A24*A32*A55+A12*A21*A34*A55-
    A11*A22*A34*A55;

    COFACTOR[4][2] = A15*A24*A32*A41-A14*A25*A32*A41-A15*A22*A34*A41+
    A12*A25*A34*A41+A14*A22*A35*A41-A12*A24*A35*A41-A15*A24*A31*A42+
    A14*A25*A31*A42+A15*A21*A34*A42-A11*A25*A34*A42-A14*A21*A35*A42+
    A11*A24*A35*A42+A15*A22*A31*A44-A12*A25*A31*A44-A15*A21*A32*A44+
    A11*A25*A32*A44+A12*A21*A35*A44-A11*A22*A35*A44-A14*A22*A31*A45+
    A12*A24*A31*A45+A14*A21*A32*A45-A11*A24*A32*A45-A12*A21*A34*A45+
    A11*A22*A34*A45;

    COFACTOR[0][3] = -A25*A33*A42*A51+A23*A35*A42*A51+A25*A32*A43*A51-
    A22*A35*A43*A51-A23*A32*A45*A51+A22*A33*A45*A51+A25*A33*A41*A52-
    A23*A35*A41*A52-A25*A31*A43*A52+A21*A35*A43*A52+A23*A31*A45*A52-
    A21*A33*A45*A52-A25*A32*A41*A53+A22*A35*A41*A53+A25*A31*A42*A53-
    A21*A35*A42*A53-A22*A31*A45*A53+A21*A32*A45*A53+A23*A32*A41*A55-
    A22*A33*A41*A55-A23*A31*A42*A55+A21*A33*A42*A55+A22*A31*A43*A55-
    A21*A32*A43*A55;

    COFACTOR[1][3] = A15*A33*A42*A51-A13*A35*A42*A51-A15*A32*A43*A51+
    A12*A35*A43*A51+A13*A32*A45*A51-A12*A33*A45*A51-A15*A33*A41*A52+
    A13*A35*A41*A52+A15*A31*A43*A52-A11*A35*A43*A52-A13*A31*A45*A52+
    A11*A33*A45*A52+A15*A32*A41*A53-A12*A35*A41*A53-A15*A31*A42*A53+
    A11*A35*A42*A53+A12*A31*A45*A53-A11*A32*A45*A53-A13*A32*A41*A55+
    A12*A33*A41*A55+A13*A31*A42*A55-A11*A33*A42*A55-A12*A31*A43*A55+
    A11*A32*A43*A55;

    COFACTOR[2][3] = -A15*A23*A42*A51+A13*A25*A42*A51+A15*A22*A43*A51-
    A12*A25*A43*A51-A13*A22*A45*A51+A12*A23*A45*A51+A15*A23*A41*A52-
    A13*A25*A41*A52-A15*A21*A43*A52+A11*A25*A43*A52+A13*A21*A45*A52-
    A11*A23*A45*A52-A15*A22*A41*A53+A12*A25*A41*A53+A15*A21*A42*A53-
    A11*A25*A42*A53-A12*A21*A45*A53+A11*A22*A45*A53+A13*A22*A41*A55-
    A12*A23*A41*A55-A13*A21*A42*A55+A11*A23*A42*A55+A12*A21*A43*A55-
    A11*A22*A43*A55;

    COFACTOR[3][3] = A15*A23*A32*A51-A13*A25*A32*A51-A15*A22*A33*A51+
    A12*A25*A33*A51+A13*A22*A35*A51-A12*A23*A35*A51-A15*A23*A31*A52+
    A13*A25*A31*A52+A15*A21*A33*A52-A11*A25*A33*A52-A13*A21*A35*A52+
    A11*A23*A35*A52+A15*A22*A31*A53-A12*A25*A31*A53-A15*A21*A32*A53+
    A11*A25*A32*A53+A12*A21*A35*A53-A11*A22*A35*A53-A13*A22*A31*A55+
    A12*A23*A31*A55+A13*A21*A32*A55-A11*A23*A32*A55-A12*A21*A33*A55+
    A11*A22*A33*A55;

    COFACTOR[4][3] = -A15*A23*A32*A41+A13*A25*A32*A41+A15*A22*A33*A41-
    A12*A25*A33*A41-A13*A22*A35*A41+A12*A23*A35*A41+A15*A23*A31*A42-
    A13*A25*A31*A42-A15*A21*A33*A42+A11*A25*A33*A42+A13*A21*A35*A42-
    A11*A23*A35*A42-A15*A22*A31*A43+A12*A25*A31*A43+A15*A21*A32*A43-
    A11*A25*A32*A43-A12*A21*A35*A43+A11*A22*A35*A43+A13*A22*A31*A45-
    A12*A23*A31*A45-A13*A21*A32*A45+A11*A23*A32*A45+A12*A21*A33*A45-
    A11*A22*A33*A45;

    COFACTOR[0][4] = A24*A33*A42*A51-A23*A34*A42*A51-A24*A32*A43*A51+
    A22*A34*A43*A51+A23*A32*A44*A51-A22*A33*A44*A51-A24*A33*A41*A52+
    A23*A34*A41*A52+A24*A31*A43*A52-A21*A34*A43*A52-A23*A31*A44*A52+
    A21*A33*A44*A52+A24*A32*A41*A53-A22*A34*A41*A53-A24*A31*A42*A53+
    A21*A34*A42*A53+A22*A31*A44*A53-A21*A32*A44*A53-A23*A32*A41*A54+
    A22*A33*A41*A54+A23*A31*A42*A54-A21*A33*A42*A54-A22*A31*A43*A54+
    A21*A32*A43*A54;

    COFACTOR[1][4] = -A14*A33*A42*A51+A13*A34*A42*A51+A14*A32*A43*A51-
    A12*A34*A43*A51-A13*A32*A44*A51+A12*A33*A44*A51+A14*A33*A41*A52-
    A13*A34*A41*A52-A14*A31*A43*A52+A11*A34*A43*A52+A13*A31*A44*A52-
    A11*A33*A44*A52-A14*A32*A41*A53+A12*A34*A41*A53+A14*A31*A42*A53-
    A11*A34*A42*A53-A12*A31*A44*A53+A11*A32*A44*A53+A13*A32*A41*A54-
    A12*A33*A41*A54-A13*A31*A42*A54+A11*A33*A42*A54+A12*A31*A43*A54-
    A11*A32*A43*A54;

    COFACTOR[2][4] = A14*A23*A42*A51-A13*A24*A42*A51-A14*A22*A43*A51+
    A12*A24*A43*A51+A13*A22*A44*A51-A12*A23*A44*A51-A14*A23*A41*A52+
    A13*A24*A41*A52+A14*A21*A43*A52-A11*A24*A43*A52-A13*A21*A44*A52+
    A11*A23*A44*A52+A14*A22*A41*A53-A12*A24*A41*A53-A14*A21*A42*A53+
    A11*A24*A42*A53+A12*A21*A44*A53-A11*A22*A44*A53-A13*A22*A41*A54+
    A12*A23*A41*A54+A13*A21*A42*A54-A11*A23*A42*A54-A12*A21*A43*A54+
    A11*A22*A43*A54;

    COFACTOR[3][4] = -A14*A23*A32*A51+A13*A24*A32*A51+A14*A22*A33*A51-
    A12*A24*A33*A51-A13*A22*A34*A51+A12*A23*A34*A51+A14*A23*A31*A52-
    A13*A24*A31*A52-A14*A21*A33*A52+A11*A24*A33*A52+A13*A21*A34*A52-
    A11*A23*A34*A52-A14*A22*A31*A53+A12*A24*A31*A53+A14*A21*A32*A53-
    A11*A24*A32*A53-A12*A21*A34*A53+A11*A22*A34*A53+A13*A22*A31*A54-
    A12*A23*A31*A54-A13*A21*A32*A54+A11*A23*A32*A54+A12*A21*A33*A54-
    A11*A22*A33*A54;

    COFACTOR[4][4] = A14*A23*A32*A41-A13*A24*A32*A41-A14*A22*A33*A41+
    A12*A24*A33*A41+A13*A22*A34*A41-A12*A23*A34*A41-A14*A23*A31*A42+
    A13*A24*A31*A42+A14*A21*A33*A42-A11*A24*A33*A42-A13*A21*A34*A42+
    A11*A23*A34*A42+A14*A22*A31*A43-A12*A24*A31*A43-A14*A21*A32*A43+
    A11*A24*A32*A43+A12*A21*A34*A43-A11*A22*A34*A43-A13*A22*A31*A44+
    A12*A23*A31*A44+A13*A21*A32*A44-A11*A23*A32*A44-A12*A21*A33*A44+
    A11*A22*A33*A44;


    for (i=0;i<dof;i++){
        for (j=0;j<dof;j++){
            Ainv[i][j]= COFACTOR[j][i]/DET;
        }
    }
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "Compute_square_root_inverse"
PetscErrorCode Compute_square_root_inverse(PetscReal (*TauTemp)[5],PetscReal (*TauTempInv)[5],PetscReal taut,void *ctx)
{

    AppCtx *user = (AppCtx *)ctx;

    //	 PetscInt  dim    = user->iga->dim;
    PetscInt  dof    = user->iga->dof;

    PetscInt i,j,aa;


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

        Invert5x5(Y, Yinv,user);

        Invert5x5(Z, Zinv,user);


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
                                          PetscReal (*A0inv)[5],PetscReal (*A1_c)[5],PetscReal (*A1_p)[5], PetscReal (*A1_pt)[5],
                                          PetscReal (*A2_c)[5],PetscReal (*A2_p)[5], PetscReal (*A2_pt)[5], PetscReal (*A3_c)[5],
                                          PetscReal (*A3_p)[5], PetscReal (*A3_pt)[5], PetscReal (*K)[3][4][4],
                                          PetscReal (*A1_cons)[5],PetscReal (*A2_cons)[5],PetscReal (*A3_cons)[5],PetscReal (*K_cons)[3][5][5],void *ctx)
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
                         PetscScalar (*tau)[5],void *ctx,PetscReal (*A0inv)[5],PetscReal (*A1_cons)[5],PetscReal (*A2_cons)[5],PetscReal (*A3_cons)[5],PetscReal (*K_cons)[3][5][5],PetscReal *umi)
{

    AppCtx *user = (AppCtx *)ctx;

    PetscInt  dof    = user->iga->dof;


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


    //Transform to pressure primitive variables


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


#undef __FUNCT__
#define __FUNCT__ "probePressureSignals"
PetscErrorCode probePressureSignals(AppCtx *user, PARAMETERS *par, ParticleManager &manager)
{
  PetscFunctionBegin;
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  PetscErrorCode ierr;

PetscReal pt[3] = {0.0};
PetscScalar uf[5] = {0.0};
IGAProbe prb;
ierr = IGAProbeCreate(user->iga,user->V1,&prb);CHKERRQ(ierr);

pt[0] = 0.25/user->Lx;
pt[1] = 0.25/user->Ly;
pt[2] = 0.304/user->Lz;

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
PetscPrintf(PETSC_COMM_WORLD, "Probe density = %e Pressure at Probe = %e\n", dens, P);
PetscPrintf(PETSC_COMM_WORLD, "u_z at Probe = %e\n", uf[3]);
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
#define __FUNCT__ "ResidualRDX"
PetscErrorCode ResidualRDX(IGAProbe prb,
                          IGAProbe prb_t,
                          IGAElement ele,
                          PetscReal dt,
                          PetscReal t,
                          PetscScalar *Re,
                          AppCtx *user)
{

    PetscErrorCode ierr;
    PetscInt  i,j,l,m;
    PetscInt  dof   = prb->dof;
    PetscInt  dim   = prb->dim;

    PetscReal lamda  = 0.0;
    PetscReal mu     = 0.0;
    PetscReal kappa  = 0.0;
    PetscReal Cv     = 143.3;
    PetscReal chi    = lamda+2*mu;

    PetscScalar u[dof];
    PetscScalar u_t[dof];
    PetscScalar grad_u[dof][dim];
    PetscReal   InvGradMap[dim][dim];

    ierr = IGAProbeFormInvGradGeomMap(prb,ele,&InvGradMap[0][0]);CHKERRQ(ierr);
    ierr = IGAProbeFormValue(prb,&u[0]);CHKERRQ(ierr);
    ierr = IGAProbeFormGrad(prb, &grad_u[0][0]);CHKERRQ(ierr);
    ierr = IGAProbeFormValue(prb_t,&u_t[0]);CHKERRQ(ierr);

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

    PetscScalar dens= u[0];
    PetscScalar ux  = u[1];
    PetscScalar uy  = u[2];
    PetscScalar uz  = u[3];
    PetscScalar temp= u[4];
    if(dens<=0.0){
      PetscPrintf(PETSC_COMM_SELF,"Density < 0, Current Explosive Volume Update error: ResidualRDX\n");
      exit(0);
    }

   PetscReal nu     = dens0/dens;
   PetscReal Ptest  = A*exp(-R1*nu) + B*exp(-R2*nu) + C/(pow(nu, 1+omega));

   PetscReal P        = Pcr;
   PetscReal fprime   = 0.0;
   PetscReal cs       = 0.0;

   if(PETSC_FALSE){//Ptest > Pcr){
     exit(0);
     P        = Ptest;
     fprime   = A*exp(-R1*nu)*R1*nu/dens +  B*exp(-R2*nu)*R2*nu/dens  +  C*(1+omega)*pow(dens,omega)/pow(dens0,1+omega);
     cs       = fprime;
   }else{
     P        = A*(1.0-omega/(R1*nu))*exp(-R1*nu) + B*(1.0-omega/(R2*nu))*exp(-R2*nu) + omega*dens0*E0/nu;

     fprime   = ((A*omega/(R1*dens0))*(R1*nu-1.0)+nu*A*R1/dens)*exp(-R1*nu) +
                ((B*omega/(R2*dens0))*(R2*nu-1.0)+nu*B*R2/dens)*exp(-R2*nu) +
                omega*E0;

     cs       = fprime;
   }

   PetscReal      umi[3] = {0.0};
   PetscScalar    tau[5][5] = {{0.0}};
   PetscReal      A1_c[5][5] = {{0.0}};
   PetscReal      A1_p[5][5] = {{0.0}};
   PetscReal      A2_c[5][5] = {{0.0}};
   PetscReal      A2_p[5][5] = {{0.0}};
   PetscReal      A3_c[5][5] = {{0.0}};
   PetscReal      A3_p[5][5] = {{0.0}};
   PetscReal      A3_pt[5][5]   = {{0.0}};
   PetscReal      A2_pt[5][5]   = {{0.0}};
   PetscReal      A1_pt[5][5]   = {{0.0}};

   PetscReal      A0[5][5] = {{0.0}};
   PetscReal      A0inv[5][5] = {{0.0}};
   PetscReal      K[3][3][4][4] = {{{{0.0}}}};
   PetscReal      G[3][3] = {{0.0}};

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

   // Density Primitive A0:
   A0[0][0] = 1.0;
   A0[1][0] = ux;
   A0[1][1] = dens;
   A0[2][0] = uy;
   A0[2][2] = dens;
   A0[3][0] = uz;
   A0[3][3] = dens;
   A0[4][0] = Cv*temp;
   A0[4][4] = dens*Cv;

   // Inverse of A0:
   A0inv[0][0] =  1.0;
   A0inv[1][0] = -ux/dens;
   A0inv[1][1] =  1.0/dens;
   A0inv[2][0] = -uy/dens;
   A0inv[2][2] =  1.0/dens;
   A0inv[3][0] = -uz/dens;
   A0inv[3][3] =  1.0/dens;
   A0inv[4][0] =  -temp/dens;
   A0inv[4][4] =  1.0/(Cv*dens);

   // Compute A1_p/adv
   A1_c[0][0] = ux;
   A1_c[0][1] = dens;
   A1_c[1][0] = ux*ux ;
   A1_c[1][1] = 2.0*dens*ux;
   A1_c[2][0] = ux*uy;
   A1_c[2][1] = dens*uy;
   A1_c[2][2] = dens*ux;
   A1_c[3][0] = ux*uz;
   A1_c[3][1] = dens*uz;
   A1_c[3][3] = dens*ux;
   A1_c[4][0] = Cv*temp*ux;
   A1_c[4][1] = dens*Cv*temp;
   A1_c[4][4] = dens*ux*Cv;

   // A_1^p
   A1_p[1][0] = fprime;

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
   A2_c[4][0] = uy*Cv*temp;
   A2_c[4][2] = dens*Cv*temp;
   A2_c[4][4] = dens*uy*Cv;

   // A_2^p
   A2_p[2][0] = fprime;

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
   A3_c[4][0] = Cv*temp*uz;
   A3_c[4][3] = Cv*temp*dens;
   A3_c[4][4] = dens*Cv*uz;

   //A_3^p
   A3_p[3][0] = fprime;

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

 PetscReal F1[5]={0.0};
 PetscReal F2[5]={0.0};
 PetscReal F3[5]={0.0};
 F1[1] = P;
 F2[2] = P;
 F3[3] = P;

 for (i=0;i<dim;i++){
 for (j=0;j<dim;j++){
 for (l=0;l<dim;l++){
 G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
 }
 }
 }

     //Direct Tau
     PetscReal A1_cons[5][5] = {{0.0}};
     PetscReal A2_cons[5][5] = {{0.0}};
     PetscReal A3_cons[5][5] = {{0.0}};
     PetscReal K_cons[3][3][5][5] = {{{{0.0}}}};

     ierr = ComputeAMatrixConservation(u, A0inv, A1_c, A1_p, A1_pt,A2_c,A2_p,A2_pt,A3_c,A3_p,A3_pt,K,A1_cons,A2_cons,A3_cons,K_cons,user);CHKERRQ(ierr);
     ierr = DirectTau(G,dt,u,tau,user,A0inv,A1_cons,A2_cons,A3_cons,K_cons,umi);CHKERRQ(ierr);

   PetscReal  *N0 = prb->shape[0];
   PetscReal (*N1)[dim] = (PetscReal (*)[dim]) prb->shape[1];
   PetscReal Res[5]     = {0.0};

   for (i=0;i<5;i++){
   for (j=0;j<5;j++){
        Res[i] += A0[i][j]*u_t[j]
               + (A1_c[i][j]+A1_p[i][j]+A1_pt[i][j])*grad_u[j][0]
               + (A2_c[i][j]+A2_p[i][j]+A2_pt[i][j])*grad_u[j][1]
               + (A3_c[i][j]+A3_p[i][j]+A3_pt[i][j])*grad_u[j][2];
   }
   }

   //Stabilization terms DC
    PetscReal hu[3] = {0.0};
    PetscReal A0gradY[5][3]={{0.0}};
    PetscReal tau_m,tau_c,tau_t;

    for (j=0;j<dim;j++){
    for (i=0;i<dof;i++){
    for (l=0;l<dof;l++){
      A0gradY[i][j] += A0[i][l]*grad_u[l][j];
    }}}

    for (i=0;i<dim;i++){
    for (j=0;j<dim;j++){
     hu[0] += A0gradY[0][i]*G[i][j]*A0gradY[0][j];
     hu[1] += A0gradY[1][i]*G[i][j]*A0gradY[1][j];
     hu[1] += A0gradY[2][i]*G[i][j]*A0gradY[2][j];
     hu[1] += A0gradY[3][i]*G[i][j]*A0gradY[3][j];
     hu[2] += A0gradY[4][i]*G[i][j]*A0gradY[4][j];
    }}

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
   PetscReal eps = 0;
   PetscReal DC = 1.0;

   /*tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs + (Res[4]*Res[4])/cs)/
   sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + hu[2]/cs+1e-15);*/
   /*tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
   sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);*/

   tau_c = sqrt((ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
   sqrt((ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);

   if(DC*tau_c > k_cap){tau_c = k_cap;}else{tau_c = DC*tau_c;}
   tau_m = tau_c;
   tau_t = tau_c;

   PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
   PetscInt a,nen=prb->nen;

   for (a=0; a<nen; a++) {
     PetscReal Na    = N0[a];
     /* ----- */
     R[a][0]  = 0.0;
     R[a][1]  = 0.0;
     R[a][2]  = 0.0;
     R[a][3]  = 0.0;
     R[a][4]  = 0.0;

     for (i=0;i<5;i++){
     for (j=0;j<5;j++){
      R[a][i] += (A0[i][j]*u_t[j]
       + (A1_c[i][j]+A1_pt[i][j])*grad_u[j][0]
       + (A2_c[i][j]+A2_pt[i][j])*grad_u[j][1]
       + (A3_c[i][j]+A3_pt[i][j])*grad_u[j][2])*Na;
     }
     }

     for (i=0;i<5;i++){
          R[a][i] += -N1[a][0]*F1[i] - N1[a][1]*F2[i] - N1[a][2]*F3[i];
     }

     for (l=0;l<dim;l++){
     for (m=0;m<dim;m++){
      for (i=0;i<dof-1;i++){
         for (j=0;j<dof-1;j++){
                R[a][i+1] += N1[a][l]*K[l][m][i][j]*grad_u[j+1][m];		//ok
         }
      }
     }
     }

 //Stabilization terms SUPG & DC
     for (i=0;i<dof;i++){
     for (j=0;j<dof;j++){
     for (l=0;l<dof;l++){
         R[a][i] +=((A1_c[i][j]+A1_p[i][j]+A1_pt[i][j])*N1[a][0]
                 + (A2_c[i][j]+A2_p[i][j]+A2_pt[i][j])*N1[a][1]
                 + (A3_c[i][j]+A3_p[i][j]+A3_pt[i][j])*N1[a][2])
                 * tau[j][l]*Res[l];
     }}}

     for (i=0;i<dim;i++)
     {
                R[a][0] +=N1[a][i]*tau_c*A0gradY[0][i];
                R[a][1] +=N1[a][i]*tau_m*A0gradY[1][i];
                R[a][2] +=N1[a][i]*tau_m*A0gradY[2][i];
                R[a][3] +=N1[a][i]*tau_m*A0gradY[3][i];
                R[a][4] +=N1[a][i]*tau_t*A0gradY[4][i];
     }
     }
   return 0;
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
     PetscScalar u[dof];

     IGAPointFormValue(pnt,U,&u[0]);
     PetscScalar dens = u[0];

     // Tait Eq. Water EOS (should be in a function)
     PetscReal dens0  = 1000.0;
     PetscReal P0     = 100000.0;
     PetscReal B      = 3.31e8;
     PetscReal N      = 7.15;
     PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
     PetscReal Pcr    = 22.02726;
     PetscReal P      = Pcr;
     PetscReal fprime = 0.0;
     if(dens>rhoCR){
     P        = P0+B*(pow(dens/dens0,N))-B;
     fprime   = (1.0/dens)*B*N*(pow(dens/dens0,N));
     }

     PetscReal F1[5]={0.0};
     PetscReal F2[5]={0.0};
     PetscReal F3[5]={0.0};
     F1[1]          = P0;//100000.0;
     F2[2]          = P0;//100000.0;
     F3[3]          = P0;//100000.0;

     PetscReal *N0 = pnt->shape[0];
     PetscReal *norm;
     PetscInt i;
     norm = pnt->normal;

     PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
     PetscInt a,nen=pnt->nen;
     for (a=0; a<nen; a++) {
       PetscReal Na = N0[a];
       for (i=0;i<5;i++)
       R[a][i] += Na*F1[i]*norm[0] + Na*F2[i]*norm[1] + Na*F3[i]*norm[2];
     }
   }

   else{ // IGA point in the interior
   PetscInt i,j,l,m;
   PetscInt  dof    = pnt->dof;
   PetscInt  dim    = pnt->dim;
   PetscReal lamda  = user->lamda;
   PetscReal mu     = user->mu;
   PetscReal kappa  = user->kappa;
   PetscReal Cv     = user->Cv;

   PetscReal dens0  = 1000.0;
   PetscReal P0     = 100000.0;
   PetscReal B      = 3.31e8;
   PetscReal N      = 7.15;
   PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
   PetscReal Pcr    = 22.02726;

   PetscScalar u[dof], u_t[dof];
   IGAPointFormValue(pnt,U,&u[0]);
   IGAPointFormValue(pnt,V,&u_t[0]);

   PetscScalar dens= u[0];
   PetscScalar ux  = u[1];
   PetscScalar uy  = u[2];
   PetscScalar uz  = u[3];
   PetscScalar temp= u[4];

   PetscReal P        = Pcr;
   PetscReal fprime   = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
   PetscReal cs       = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
   if(dens>rhoCR){
      P        = P0+B*(pow(dens/dens0,N))-B;
      fprime   = (1.0/dens)*B*N*(pow(dens/dens0,N));
      cs       = (1.0/dens)*B*N*(pow(dens/dens0,N));}
   PetscReal   chi    = lamda+2*mu;

  PetscScalar grad_u[dof][dim];
  IGAPointFormGrad (pnt,U,&grad_u[0][0]);
  PetscReal InvGradMap[dim][dim];
  IGAPointFormInvGradGeomMap(pnt,&InvGradMap[0][0]);

  PetscReal      umi[3] = {0.0};
  PetscScalar    tau[5][5] = {{0.0}};
  PetscReal      A1_c[5][5] = {{0.0}};
  PetscReal      A1_p[5][5] = {{0.0}};
  PetscReal      A2_c[5][5] = {{0.0}};
  PetscReal      A2_p[5][5] = {{0.0}};
  PetscReal      A3_c[5][5] = {{0.0}};
  PetscReal      A3_p[5][5] = {{0.0}};
  PetscReal      A3_pt[5][5]   = {{0.0}};
  PetscReal      A2_pt[5][5]   = {{0.0}};
  PetscReal      A1_pt[5][5]   = {{0.0}};
  PetscReal      A0[5][5] = {{0.0}};
  PetscReal      A0inv[5][5] = {{0.0}};
  PetscReal      K[3][3][4][4] = {{{{0.0}}}};
  PetscReal      G[3][3] = {{0.0}};

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

  // Density Primitive A0:
  A0[0][0] = 1.0;
  A0[1][0] = ux;
  A0[1][1] = dens;
  A0[2][0] = uy;
  A0[2][2] = dens;
  A0[3][0] = uz;
  A0[3][3] = dens;
  A0[4][0] = Cv*temp;
  A0[4][4] = dens*Cv;

  // Inverse of A0:
  A0inv[0][0] =  1.0;
  A0inv[1][0] = -ux/dens;
  A0inv[1][1] =  1.0/dens;
  A0inv[2][0] = -uy/dens;
  A0inv[2][2] =  1.0/dens;
  A0inv[3][0] = -uz/dens;
  A0inv[3][3] =  1.0/dens;
  A0inv[4][0] =  -temp/dens;
  A0inv[4][4] =  1.0/(Cv*dens);

  // Compute A1_p/adv
  A1_c[0][0] = ux;
  A1_c[0][1] = dens;
  A1_c[1][0] = ux*ux ;
  A1_c[1][1] = 2.0*dens*ux;
  A1_c[2][0] = ux*uy;
  A1_c[2][1] = dens*uy;
  A1_c[2][2] = dens*ux;
  A1_c[3][0] = ux*uz;
  A1_c[3][1] = dens*uz;
  A1_c[3][3] = dens*ux;
  A1_c[4][0] = Cv*temp*ux;
  A1_c[4][1] = dens*Cv*temp;
  A1_c[4][4] = dens*ux*Cv;

  // A_1^p
  A1_p[1][0] = fprime;

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
  A2_c[4][0] = uy*Cv*temp;
  A2_c[4][2] = dens*Cv*temp;
  A2_c[4][4] = dens*uy*Cv;

  // A_2^p
  A2_p[2][0] = fprime;

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
  A3_c[4][0] = Cv*temp*uz;
  A3_c[4][3] = Cv*temp*dens;
  A3_c[4][4] = dens*Cv*uz;

  //A_3^p
  A3_p[3][0] = fprime;

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

PetscReal F1[5]={0.0};
PetscReal F2[5]={0.0};
PetscReal F3[5]={0.0};
F1[1] = P;
F2[2] = P;
F3[3] = P;

for (i=0;i<dim;i++){
for (j=0;j<dim;j++){
for (l=0;l<dim;l++){
G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
}
}
}

    //Direct Tau
    PetscReal A1_cons[5][5] = {{0.0}};
    PetscReal A2_cons[5][5] = {{0.0}};
    PetscReal A3_cons[5][5] = {{0.0}};
    PetscReal K_cons[3][3][5][5] = {{{{0.0}}}};

    ierr = ComputeAMatrixConservation(u, A0inv, A1_c, A1_p, A1_pt,A2_c,A2_p,A2_pt,A3_c,A3_p,A3_pt,K,A1_cons,A2_cons,A3_cons,K_cons,user);CHKERRQ(ierr);
    ierr = DirectTau(G,dt,u,tau,user,A0inv,A1_cons,A2_cons,A3_cons,K_cons,umi);CHKERRQ(ierr);

  PetscReal  *N0       = pnt->shape[0];
  PetscReal (*N1)[dim] = (PetscReal (*)[dim]) pnt->shape[1];
  PetscReal Res[5]     = {0.0};

  for (i=0;i<5;i++){
  for (j=0;j<5;j++){
        Res[i] += A0[i][j]*u_t[j] + (A1_c[i][j]+A1_p[i][j]+A1_pt[i][j])*grad_u[j][0]
        + (A2_c[i][j]+A2_p[i][j]+A2_pt[i][j])*grad_u[j][1] + (A3_c[i][j]+A3_p[i][j]+A3_pt[i][j])*grad_u[j][2];
  }
  }

  //Stabilization terms DC
   PetscReal hu[3] = {0.0};
   PetscReal A0gradY[5][3]={{0.0}};
   PetscReal tau_m,tau_c,tau_t;

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
  PetscReal DC = 1.0;

  // tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs + (Res[4]*Res[4])/cs)/
  // sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + hu[2]/cs+1e-15);
  /*tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
  sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);*/

  tau_c = sqrt((ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
  sqrt((ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);

  if(DC*tau_c > k_cap){tau_c = k_cap;}else{tau_c = DC*tau_c;}
  tau_m = tau_c;
  tau_t = tau_c;

  PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
  PetscInt a,nen=pnt->nen;

  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    /* ----- */
    R[a][0]  = 0.0;
    R[a][1]  = 0.0;
    R[a][2]  = 0.0;
    R[a][3]  = 0.0;
    R[a][4]  = 0.0;

    for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      R[a][i] += (A0[i][j]*u_t[j]
      + (A1_c[i][j]+A1_pt[i][j])*grad_u[j][0]
      + (A2_c[i][j]+A2_pt[i][j])*grad_u[j][1]
      + (A3_c[i][j]+A3_pt[i][j])*grad_u[j][2])*Na;
    }
    }

    for (i=0;i<5;i++){
          R[a][i] += -N1[a][0]*F1[i] - N1[a][1]*F2[i] - N1[a][2]*F3[i];
    }

    for (l=0;l<dim;l++){
    for (m=0;m<dim;m++){
      for (i=0;i<dof-1;i++){
        for (j=0;j<dof-1;j++){
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
    }
    }
   }
   return 0;
 }

 #undef  __FUNCT__
 #define __FUNCT__ "ResidualFS"
 PetscErrorCode ResidualFS(IGAProbe prb,
                           IGAProbe prb_t,
                           IGAElement ele,
                         PetscReal dt,
                         PetscReal t,
                         PetscScalar *Re,
                         void *ctx)
 {
   PetscErrorCode ierr;
   AppCtx *user = (AppCtx *)ctx;

   PetscInt i,j,l,m;
   PetscInt  dof    = prb->dof;
   PetscInt  dim    = prb->dim;
   PetscReal lamda  = user->lamda;
   PetscReal mu     = user->mu;
   PetscReal kappa  = user->kappa;
   PetscReal Cv     = user->Cv;

   PetscReal dens0  = 1000.0;
   PetscReal P0     = 100000.0;
   PetscReal B      = 3.31e8;
   PetscReal N      = 7.15;
   PetscReal rhoCR  = dens0*pow((1/B)*(22.02726-P0)+1, 1/N);
   PetscReal Pcr    = 22.02726;

   PetscScalar u[dof];
   PetscScalar u_t[dof];
   PetscScalar grad_u[dof][dim];
   PetscReal   InvGradMap[dim][dim];

   ierr = IGAProbeFormValue(prb,&u[0]);CHKERRQ(ierr);
   ierr = IGAProbeFormGrad(prb, &grad_u[0][0]);CHKERRQ(ierr);
   ierr = IGAProbeFormValue(prb_t,&u_t[0]);CHKERRQ(ierr);
   ierr = IGAProbeFormInvGradGeomMap(prb,ele,&InvGradMap[0][0]);CHKERRQ(ierr);

   PetscScalar dens= u[0];
   PetscScalar ux  = u[1];
   PetscScalar uy  = u[2];
   PetscScalar uz  = u[3];
   PetscScalar temp= u[4];

   PetscReal P        = Pcr;
   PetscReal fprime   = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
   PetscReal cs       = (1.0/rhoCR)*B*N*(pow(rhoCR/dens0,N));
   if(dens>rhoCR){
      P        = P0+B*(pow(dens/dens0,N))-B;
      fprime   = (1.0/dens)*B*N*(pow(dens/dens0,N));
      cs       = (1.0/dens)*B*N*(pow(dens/dens0,N));}
   PetscReal   chi    = lamda+2*mu;

  PetscReal      umi[3] = {0.0};
  PetscScalar    tau[5][5] = {{0.0}};
  PetscReal      A1_c[5][5] = {{0.0}};
  PetscReal      A1_p[5][5] = {{0.0}};
  PetscReal      A2_c[5][5] = {{0.0}};
  PetscReal      A2_p[5][5] = {{0.0}};
  PetscReal      A3_c[5][5] = {{0.0}};
  PetscReal      A3_p[5][5] = {{0.0}};
  PetscReal      A3_pt[5][5]   = {{0.0}};
  PetscReal      A2_pt[5][5]   = {{0.0}};
  PetscReal      A1_pt[5][5]   = {{0.0}};
  PetscReal      A0[5][5] = {{0.0}};
  PetscReal      A0inv[5][5] = {{0.0}};
  PetscReal      K[3][3][4][4] = {{{{0.0}}}};
  PetscReal      G[3][3] = {{0.0}};

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

  // Density Primitive A0:
  A0[0][0] = 1.0;
  A0[1][0] = ux;
  A0[1][1] = dens;
  A0[2][0] = uy;
  A0[2][2] = dens;
  A0[3][0] = uz;
  A0[3][3] = dens;
  A0[4][0] = Cv*temp;
  A0[4][4] = dens*Cv;

  // Inverse of A0:
  A0inv[0][0] =  1.0;
  A0inv[1][0] = -ux/dens;
  A0inv[1][1] =  1.0/dens;
  A0inv[2][0] = -uy/dens;
  A0inv[2][2] =  1.0/dens;
  A0inv[3][0] = -uz/dens;
  A0inv[3][3] =  1.0/dens;
  A0inv[4][0] =  -temp/dens;
  A0inv[4][4] =  1.0/(Cv*dens);

  // Compute A1_p/adv
  A1_c[0][0] = ux;
  A1_c[0][1] = dens;
  A1_c[1][0] = ux*ux ;
  A1_c[1][1] = 2.0*dens*ux;
  A1_c[2][0] = ux*uy;
  A1_c[2][1] = dens*uy;
  A1_c[2][2] = dens*ux;
  A1_c[3][0] = ux*uz;
  A1_c[3][1] = dens*uz;
  A1_c[3][3] = dens*ux;
  A1_c[4][0] = Cv*temp*ux;
  A1_c[4][1] = dens*Cv*temp;
  A1_c[4][4] = dens*ux*Cv;

  // A_1^p
  A1_p[1][0] = fprime;

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
  A2_c[4][0] = uy*Cv*temp;
  A2_c[4][2] = dens*Cv*temp;
  A2_c[4][4] = dens*uy*Cv;

  // A_2^p
  A2_p[2][0] = fprime;

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
  A3_c[4][0] = Cv*temp*uz;
  A3_c[4][3] = Cv*temp*dens;
  A3_c[4][4] = dens*Cv*uz;

  //A_3^p
  A3_p[3][0] = fprime;

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

PetscReal F1[5]={0.0};
PetscReal F2[5]={0.0};
PetscReal F3[5]={0.0};
F1[1] = P;
F2[2] = P;
F3[3] = P;

for (i=0;i<dim;i++){
for (j=0;j<dim;j++){
for (l=0;l<dim;l++){
G[i][j] += InvGradMap[i][l]*InvGradMap[j][l];
}
}
}

    //Direct Tau
    PetscReal A1_cons[5][5] = {{0.0}};
    PetscReal A2_cons[5][5] = {{0.0}};
    PetscReal A3_cons[5][5] = {{0.0}};
    PetscReal K_cons[3][3][5][5] = {{{{0.0}}}};

    ierr = ComputeAMatrixConservation(u, A0inv, A1_c, A1_p, A1_pt,A2_c,A2_p,A2_pt,A3_c,A3_p,A3_pt,K,A1_cons,A2_cons,A3_cons,K_cons,user);CHKERRQ(ierr);
    ierr = DirectTau(G,dt,u,tau,user,A0inv,A1_cons,A2_cons,A3_cons,K_cons,umi);CHKERRQ(ierr);

  PetscReal  *N0       = prb->shape[0];
  PetscReal (*N1)[dim] = (PetscReal (*)[dim]) prb->shape[1];
  PetscReal Res[5]     = {0.0};

  for (i=0;i<5;i++){
  for (j=0;j<5;j++){
        Res[i] += A0[i][j]*u_t[j] + (A1_c[i][j]+A1_p[i][j]+A1_pt[i][j])*grad_u[j][0]
        + (A2_c[i][j]+A2_p[i][j]+A2_pt[i][j])*grad_u[j][1] + (A3_c[i][j]+A3_p[i][j]+A3_pt[i][j])*grad_u[j][2];
  }
  }

  //Stabilization terms DC
   PetscReal hu[3] = {0.0};
   PetscReal A0gradY[5][3]={{0.0}};
   PetscReal tau_m,tau_c,tau_t;

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
  PetscReal DC = 1.0;

  // tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs + (Res[4]*Res[4])/cs)/
  // sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + hu[2]/cs+1e-15);
  /*tau_c = sqrt(cs*Res[0]*Res[0] + (ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
  sqrt(cs*hu[0] + (ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);*/

  tau_c = sqrt((ux*ux+uy*uy+uz*uz)*(Res[1]*Res[1]+Res[2]*Res[2]+Res[3]*Res[3])/cs)/
  sqrt((ux*ux+uy*uy+uz*uz)*hu[1]/cs + 1e-15);

  if(DC*tau_c > k_cap){tau_c = k_cap;}else{tau_c = DC*tau_c;}
  tau_m = tau_c;
  tau_t = tau_c;

  PetscScalar (*R)[dof] = (PetscScalar (*)[dof])Re;
  PetscInt a,nen=prb->nen;

  for (a=0; a<nen; a++) {
    PetscReal Na    = N0[a];
    /* ----- */
    R[a][0]  = 0.0;
    R[a][1]  = 0.0;
    R[a][2]  = 0.0;
    R[a][3]  = 0.0;
    R[a][4]  = 0.0;

    for (i=0;i<5;i++){
    for (j=0;j<5;j++){
      R[a][i] += (A0[i][j]*u_t[j]
      + (A1_c[i][j]+A1_pt[i][j])*grad_u[j][0]
      + (A2_c[i][j]+A2_pt[i][j])*grad_u[j][1]
      + (A3_c[i][j]+A3_pt[i][j])*grad_u[j][2])*Na;
    }
    }

    for (i=0;i<5;i++){
          R[a][i] += -N1[a][0]*F1[i] - N1[a][1]*F2[i] - N1[a][2]*F3[i];
    }

    for (l=0;l<dim;l++){
    for (m=0;m<dim;m++){
      for (i=0;i<dof-1;i++){
        for (j=0;j<dof-1;j++){
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
    }
    }

   return 0;
 }
//// End Residual (Fluid) Definitions ////


//// Mass Matricies ////
#undef  __FUNCT__
#define __FUNCT__ "IGAComputeMassFS"
PetscErrorCode IGAComputeMassFS(PARAMETERS *par,
                                PetscReal shift,
                                Vec U,
                                AppCtx *user,
                                Mat MassFS,
                                ParticleManager &manager)
{
// Output MassFS, mass matrix difference between fluid mass mat and immersed fluid (RDX) mass mat on the immersed particle
// fluid domain

  PetscFunctionBegin;
  PetscErrorCode ierr;

  ierr = MatZeroEntries(MassFS);CHKERRQ(ierr);

  IGAElement ele;
  ierr = IGAElementCreate(&ele);CHKERRQ(ierr);
  ierr = IGAElementInit(ele,user->iga);CHKERRQ(ierr);
  PetscInt dof = user->iga->dof;
  PetscScalar u[dof];
  IGAProbe prb;
  ierr = IGAProbeCreate(user->iga,U,&prb);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(prb, PETSC_FALSE);CHKERRQ(ierr);


  pair<OutEdgeIterator,OutEdgeIterator> its
    = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){

    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);

  if((!info.isTask) && (fd.Inside == 1) && fd.material!=0){

			PetscReal q[3];
			q[0] = info.currentCoord[0]/user->Lx;
			q[1] = info.currentCoord[1]/user->Ly;
			q[2] = info.currentCoord[2]/user->Lz;

      if(IGALocateElement_1(user->iga, q, ele)){
      ierr = IGAProbeSetPoint(prb,q);CHKERRQ(ierr);
      ierr = IGAProbeFormValue(prb,&u[0]);CHKERRQ(ierr);

			PetscReal *N0 = prb->shape[0];
			PetscInt  nen = prb->nen;
			PetscInt  dof = prb->dof;
			PetscInt  a;
		  PetscReal Cv  = user->Cv;

			PetscReal A0[5][5]={{0.0}};
			PetscReal A01[5][5]={{0.0}};

			PetscScalar dens   = u[0];
			PetscScalar ux     = u[1];
			PetscScalar uy     = u[2];
			PetscScalar uz     = u[3];
			PetscScalar temp   = u[4];

      // A0_water (immersing fluid) //
      A0[0][0] = 1.0;
      A0[1][0] = ux;
      A0[1][1] = dens;
      A0[2][0] = uy;
      A0[2][2] = dens;
      A0[3][0] = uz;
      A0[3][3] = dens;
      A0[4][0] = Cv*temp;
      A0[4][4] = dens*Cv;

      if(fd.material==1){
        //A0_RDX (new Cv required only: this is carried from TNT calculation,
       //change eventually)
        Cv    = 143.3;
        A01[0][0] = 1.0;
        A01[1][0] = ux;
        A01[1][1] = dens;
        A01[2][0] = uy;
        A01[2][2] = dens;
        A01[3][0] = uz;
        A01[3][3] = dens;
        A01[4][0] = Cv*temp;
        A01[4][4] = dens*Cv;
      }

				PetscScalar KK[dof*dof];
				PetscScalar KK1[dof*dof];

				for (a=0; a<nen; a++) {

					KK[0] = A0[0][0]*shift*N0[a]*fd.nodalVolume;
					KK[1] = A0[0][1]*shift*N0[a]*fd.nodalVolume;
					KK[2] = A0[0][2]*shift*N0[a]*fd.nodalVolume;
					KK[3] = A0[0][3]*shift*N0[a]*fd.nodalVolume;
					KK[4] = A0[0][4]*shift*N0[a]*fd.nodalVolume;

					KK[5] = A0[1][0]*shift*N0[a]*fd.nodalVolume;
					KK[6] = A0[1][1]*shift*N0[a]*fd.nodalVolume;
					KK[7] = A0[1][2]*shift*N0[a]*fd.nodalVolume;
					KK[8] = A0[1][3]*shift*N0[a]*fd.nodalVolume;
					KK[9] = A0[1][4]*shift*N0[a]*fd.nodalVolume;

					KK[10] = A0[2][0]*shift*N0[a]*fd.nodalVolume;
					KK[11] = A0[2][1]*shift*N0[a]*fd.nodalVolume;
					KK[12] = A0[2][2]*shift*N0[a]*fd.nodalVolume;
					KK[13] = A0[2][3]*shift*N0[a]*fd.nodalVolume;
					KK[14] = A0[2][4]*shift*N0[a]*fd.nodalVolume;

					KK[15] = A0[3][0]*shift*N0[a]*fd.nodalVolume;
					KK[16] = A0[3][1]*shift*N0[a]*fd.nodalVolume;
					KK[17] = A0[3][2]*shift*N0[a]*fd.nodalVolume;
					KK[18] = A0[3][3]*shift*N0[a]*fd.nodalVolume;
					KK[19] = A0[3][4]*shift*N0[a]*fd.nodalVolume;

					KK[20] = A0[4][0]*shift*N0[a]*fd.nodalVolume;
					KK[21] = A0[4][1]*shift*N0[a]*fd.nodalVolume;
					KK[22] = A0[4][2]*shift*N0[a]*fd.nodalVolume;
					KK[23] = A0[4][3]*shift*N0[a]*fd.nodalVolume;
					KK[24] = A0[4][4]*shift*N0[a]*fd.nodalVolume;

					KK1[0] = A01[0][0]*shift*N0[a]*fd.nodalVolume;
					KK1[1] = A01[0][1]*shift*N0[a]*fd.nodalVolume;
					KK1[2] = A01[0][2]*shift*N0[a]*fd.nodalVolume;
					KK1[3] = A01[0][3]*shift*N0[a]*fd.nodalVolume;
					KK1[4] = A01[0][4]*shift*N0[a]*fd.nodalVolume;

					KK1[5] = A01[1][0]*shift*N0[a]*fd.nodalVolume;
					KK1[6] = A01[1][1]*shift*N0[a]*fd.nodalVolume;
					KK1[7] = A01[1][2]*shift*N0[a]*fd.nodalVolume;
					KK1[8] = A01[1][3]*shift*N0[a]*fd.nodalVolume;
					KK1[9] = A01[1][4]*shift*N0[a]*fd.nodalVolume;

					KK1[10] = A01[2][0]*shift*N0[a]*fd.nodalVolume;
					KK1[11] = A01[2][1]*shift*N0[a]*fd.nodalVolume;
					KK1[12] = A01[2][2]*shift*N0[a]*fd.nodalVolume;
					KK1[13] = A01[2][3]*shift*N0[a]*fd.nodalVolume;
					KK1[14] = A01[2][4]*shift*N0[a]*fd.nodalVolume;

					KK1[15] = A01[3][0]*shift*N0[a]*fd.nodalVolume;
					KK1[16] = A01[3][1]*shift*N0[a]*fd.nodalVolume;
					KK1[17] = A01[3][2]*shift*N0[a]*fd.nodalVolume;
					KK1[18] = A01[3][3]*shift*N0[a]*fd.nodalVolume;
					KK1[19] = A01[3][4]*shift*N0[a]*fd.nodalVolume;

					KK1[20] = A01[4][0]*shift*N0[a]*fd.nodalVolume;
					KK1[21] = A01[4][1]*shift*N0[a]*fd.nodalVolume;
					KK1[22] = A01[4][2]*shift*N0[a]*fd.nodalVolume;
					KK1[23] = A01[4][3]*shift*N0[a]*fd.nodalVolume;
					KK1[24] = A01[4][4]*shift*N0[a]*fd.nodalVolume;

					KK[0] -= KK1[0];
					KK[1] -= KK1[1];
					KK[2] -= KK1[2];
					KK[3] -= KK1[3];
					KK[4] -= KK1[4];
					KK[5] -= KK1[5];
					KK[6] -= KK1[6];
					KK[7] -= KK1[7];
					KK[8] -= KK1[8];
					KK[9] -= KK1[9];
					KK[10] -= KK1[10];
					KK[11] -= KK1[11];
					KK[12] -= KK1[12];
					KK[13] -= KK1[13];
					KK[14] -= KK1[14];
					KK[15] -= KK1[15];
					KK[16] -= KK1[16];
					KK[17] -= KK1[17];
					KK[18] -= KK1[18];
					KK[19] -= KK1[19];
					KK[20] -= KK1[20];
					KK[21] -= KK1[21];
					KK[22] -= KK1[22];
					KK[23] -= KK1[23];
					KK[24] -= KK1[24];

					PetscInt ID = ele->mapping[a];
					PetscInt index_array[5] = {0};
					index_array[0] = ID*dof;
					index_array[1] = ID*dof + 1;
					index_array[2] = ID*dof + 2;
					index_array[3] = ID*dof + 3;
					index_array[4] = ID*dof + 4;

					MatSetValuesLocal(MassFS,5,index_array,5,index_array,KK,ADD_VALUES);CHKERRQ(ierr);
					}
        }
      }
    }
  ierr = MatAssemblyBegin(MassFS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(MassFS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = IGAElementDestroy(&ele);CHKERRQ(ierr);
  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
////////////////////////

//// Tangent Integrands ////
#undef  __FUNCT__
#define __FUNCT__ "Tangent"
PetscErrorCode Tangent(IGAPoint pnt,
                       PetscReal shift,const PetscScalar *V,
                       PetscReal t,const PetscScalar *U,
                       PetscScalar *Ke,void *ctx)
{
// Tangent matrix KK_{ajal} = A0_{jl}*shift*N_a, immersing fluid
  AppCtx      *user = (AppCtx *)ctx;
  PetscInt 	  dof   = pnt->dof;
  PetscReal   Cv    = user->Cv;
  PetscScalar u[dof];

  IGAPointFormValue(pnt,U,&u[0]);
  PetscReal *N0 = pnt->shape[0];
  PetscReal  A0[5][5]={{0.0}};

  PetscScalar dens= u[0];
  PetscScalar ux  = u[1];
  PetscScalar uy  = u[2];
  PetscScalar uz  = u[3];
  PetscScalar temp= u[4];

  // Density Primitive A0:
  A0[0][0] = 1.0;
  A0[1][0] = ux;
  A0[1][1] = dens;
  A0[2][0] = uy;
  A0[2][2] = dens;
  A0[3][0] = uz;
  A0[3][3] = dens;
  A0[4][0] = Cv*temp;
  A0[4][4] = dens*Cv;

  PetscInt a,nen=pnt->nen;
  PetscScalar (*KK)[dof][nen][dof] = (PetscScalar (*)[dof][nen][dof])Ke;
  for (a=0; a<nen; a++) {
      KK[a][0][a][0] = A0[0][0]*shift*N0[a];
      KK[a][0][a][1] = A0[0][1]*shift*N0[a];
      KK[a][0][a][2] = A0[0][2]*shift*N0[a];
      KK[a][0][a][3] = A0[0][3]*shift*N0[a];
      KK[a][0][a][4] = A0[0][4]*shift*N0[a];

      KK[a][1][a][0] = A0[1][0]*shift*N0[a];
      KK[a][1][a][1] = A0[1][1]*shift*N0[a];
      KK[a][1][a][2] = A0[1][2]*shift*N0[a];
      KK[a][1][a][3] = A0[1][3]*shift*N0[a];
      KK[a][1][a][4] = A0[1][4]*shift*N0[a];

      KK[a][2][a][0] = A0[2][0]*shift*N0[a];
      KK[a][2][a][1] = A0[2][1]*shift*N0[a];
      KK[a][2][a][2] = A0[2][2]*shift*N0[a];
      KK[a][2][a][3] = A0[2][3]*shift*N0[a];
      KK[a][2][a][4] = A0[2][4]*shift*N0[a];

      KK[a][3][a][0] = A0[3][0]*shift*N0[a];
      KK[a][3][a][1] = A0[3][1]*shift*N0[a];
      KK[a][3][a][2] = A0[3][2]*shift*N0[a];
      KK[a][3][a][3] = A0[3][3]*shift*N0[a];
      KK[a][3][a][4] = A0[3][4]*shift*N0[a];

      KK[a][4][a][0] = A0[4][0]*shift*N0[a];
      KK[a][4][a][1] = A0[4][1]*shift*N0[a];
      KK[a][4][a][2] = A0[4][2]*shift*N0[a];
      KK[a][4][a][3] = A0[4][3]*shift*N0[a];
      KK[a][4][a][4] = A0[4][4]*shift*N0[a];
  }
  return 0;
}
//////////////////////////

//// Integration functions ////
#undef  __FUNCT__
#define __FUNCT__ "IGAComputeFS"
PetscErrorCode IGAComputeFS(IGA iga,
                            PARAMETERS *par,
                            PetscReal dt,
                            PetscReal t,
                            Vec V,
                            Vec U,
                            AppCtx *user,
                            Vec vecF,
                            ParticleManager &manager)
{
  PetscErrorCode ierr;
  ierr = VecZeroEntries(vecF);CHKERRQ(ierr);

  PetscInt i, j, a;
  PetscReal pt[3] = {0.0};

  PetscInt nen = user->nen;
  PetscInt dof = user->iga->dof;
  PetscInt dim = user->iga->dim;

  PetscScalar Kvec[nen*dof];
  PetscScalar Kvec1[nen*dof];

  PetscReal nodalVolume;
  IGAProbe prb;
  IGAProbe prb_t;
  IGAElement ele1;

  ierr = IGAProbeCreate(user->iga,U,&prb);CHKERRQ(ierr);
  ierr = IGAProbeCreate(iga,V,&prb_t);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(prb, PETSC_FALSE);CHKERRQ(ierr);
  ierr = IGAProbeSetCollective(prb_t, PETSC_FALSE);CHKERRQ(ierr);

  pair<OutEdgeIterator,OutEdgeIterator> its
    = out_edges(manager.myTaskVertex,manager.graph);
  for(auto it=its.first; it != its.second; ++it){
    ierr = IGAElementCreate(&ele1);CHKERRQ(ierr);
    ierr = IGAElementInit(ele1,user->iga);CHKERRQ(ierr);

    ele1->nval = 0;
    Edge edge = *it;
    Vertex v = target(edge,manager.graph);
    ParticleInfo info = get(info_property,v);
    FieldData fd = get(fd_property,v);
    if((!info.isTask) && (fd.Inside == 1) && fd.material!=0){

    pt[0] = info.currentCoord[0]/user->Lx;
    pt[1] = info.currentCoord[1]/user->Ly;
    pt[2] = info.currentCoord[2]/user->Lz;
    nodalVolume = fd.nodalVolume;

  if(IGALocateElement_1(user->iga,pt,ele1)){
    ierr = IGAProbeSetPoint(prb,pt);CHKERRQ(ierr);
    ierr = IGAProbeSetPoint(prb_t,pt);CHKERRQ(ierr);

    ierr = ResidualFS(prb, prb_t, ele1, dt, t, Kvec, user);CHKERRQ(ierr);
    if(fd.material == 1){
    ierr = ResidualRDX(prb, prb_t, ele1, dt, t, Kvec1, user);CHKERRQ(ierr);
    }

    for (a=0;a<nen;a++){
      PetscInt GlobalID = prb->map[a];
      for(j=0; j< dof; j++){

        Kvec[a*dof+j] *= nodalVolume;
        Kvec1[a*dof+j] *= nodalVolume;

        // Subtracting over PD solid volume, omitted for penalty coupled shells
        // if(fd.material==0){
        //   Kvec[a*dof+j] = -1.0*Kvec[a*dof+j];
        //   ierr = VecSetValueLocal(vecF,GlobalID*dof+j,Kvec[a*dof+j],ADD_VALUES);CHKERRQ(ierr);
        // }
        if(fd.material==1 || fd.material==2){
          Kvec1[a*dof+j] -= Kvec[a*dof+j];
          ierr = VecSetValueLocal(vecF,GlobalID*dof+j,Kvec1[a*dof+j],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
}

  ierr = IGAElementDestroy(&ele1);CHKERRQ(ierr);
  }
  ierr = IGAProbeDestroy(&prb);CHKERRQ(ierr);
  ierr = IGAProbeDestroy(&prb_t);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(vecF);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecF);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
///////////////////////////////

//// Generalized alpha Time Integration + Lagrangian update routines ////
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
        // if(i==2){
        //   velDiff[i] = -fd.interpolatedVelocity[i]; // The plate cannot move in the z-direction on this domain.
        // }
      }

    for(i=0; i<user->iga->dim; i++){
      if(PETSC_TRUE){
        fd.residual[i] += fd.penaltyParameter * (velDiff[0]*fd.normal[0] + velDiff[1]*fd.normal[1] + velDiff[2]*fd.normal[2])*fd.normal[i];
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
      // if(i==2){
      //   velDiff[i] = fd.interpolatedVelocity[i]; // The plate cannot move in the z-direction on this domain.
      // }
    }

    for (a=0;a<user->nen;a++){
      PetscInt GlobalID = prb->map[a];

      for(i=0; i<dim; i++){

        value = fd.penaltyParameter
         * (prb->shape[0][a]
         * (velDiff[0]*fd.normal[0] + velDiff[1]*fd.normal[1] + velDiff[2]*fd.normal[2])*fd.normal[i])
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
    cout << "\n--Peridigm-PD Shells + ImmersedIGA Compressible flow (Penalty Coupling Approach)--" << endl ;
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
  user.thickness = 0.00126;

  /* Set discretization options */
  PetscInt p=2, C=PETSC_DECIDE;

  // MPI initialization
  boost::mpi::environment env(argc,argv);

  // create an empty particle manager
  ParticleManager manager = ParticleManager();

  // seed random number generator based on MPI rank
  //srand(manager.myRank);

  par->initialTime  	   = 0.0;
  par->finalTime    	   = 4.0e-3;
  par->timeStep	         = 0.1e-6;
  par->currentTime  	   = par->initialTime;
  par->gamma			       = 0.5;
  par->stepNumber        = 0;
  par->FreqResults       = 10;
  par->densityRDX        = 1770.0;
  par->density           = 1420.0;//7.8e3; //Properties of composite plate
  par->youngModulus      = 78.24e9;//200e9;
  par->poissonRatio      = 0.0385;//0.3;

  // Parameters for Penalty based approach
  par->penaltyConstant = 6.0;
  par->DamageModeling = PETSC_FALSE;
  par->damageCriticalStress = 8.0e10;
  par->damageCriticalEpsilonPlastic = 0.2;
  par->thresholdDamageForPenalty = 0.9;

  user.TimeRestart       = 0;
  user.StepRestart       = 0;
  user.FreqRestarts      = 100;

  user.spacing  = 0.0000000000000000;
  user.Lx       = 0.5+user.spacing;
  user.Ly       = 0.5+user.spacing;
  user.Lz       = 0.5+user.spacing;
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

  //IGA
  int RestartStep = round(user.OutputRestart/par->timeStep);
  PetscInt dim = 3;
  IGA iga;
  ierr = IGACreate(PETSC_COMM_WORLD,&iga);CHKERRQ(ierr);
  ierr = IGASetDim(iga,dim);CHKERRQ(ierr);
  ierr = IGASetDof(iga,5);CHKERRQ(ierr);
  ierr = IGAGetDim(iga,&dim);CHKERRQ(ierr);
  ierr = IGARead(iga,"./Geo/Geometry.dat");CHKERRQ(ierr);
  ierr = IGASetUp(iga);CHKERRQ(ierr);
  ierr = IGAWrite(iga,"igaF.dat");CHKERRQ(ierr);
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

  ierr = MPI_Allreduce(&tempX,user.processor_numElX,iga->proc_sizes[0],MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&tempY,user.processor_numElY,iga->proc_sizes[1],MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&tempZ,user.processor_numElZ,iga->proc_sizes[2],MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);

  for (j=0;j<iga->proc_sizes[0];j++) user.processor_numElX[j] /= iga->proc_sizes[1]*iga->proc_sizes[2];
  for (j=0;j<iga->proc_sizes[1];j++) user.processor_numElY[j] /= iga->proc_sizes[0]*iga->proc_sizes[2];
  for (j=0;j<iga->proc_sizes[2];j++) user.processor_numElZ[j] /= iga->proc_sizes[0]*iga->proc_sizes[1];

  user.H[0] = user.Lx/iga->axis[0]->nel;
  user.H[1] = user.Ly/iga->axis[1]->nel;
  user.H[2] = user.Lz/iga->axis[2]->nel;

  //Creating vector for lumped mass vector. Strictly speaking it should be a matrix,
  //but since it is diagonal, we save it as a vector.
  Vec Mass;
  ierr = IGACreateVec(iga,&Mass);CHKERRQ(ierr);
  ierr = VecZeroEntries(Mass);CHKERRQ(ierr);

  Mat MassFS;
  ierr = IGACreateMat(iga,&MassFS);CHKERRQ(ierr);
  ierr = MatZeroEntries(MassFS);CHKERRQ(ierr);

  Vec ResF;
  ierr = IGACreateVec(iga,&ResF);CHKERRQ(ierr);
  ierr = VecZeroEntries(ResF);CHKERRQ(ierr);

  Vec ResFS;
  ierr = IGACreateVec(iga,&ResFS);CHKERRQ(ierr);
  ierr = VecZeroEntries(ResFS);CHKERRQ(ierr);

  // This comes from PD force state and background kinematcs
  Vec ResS;
  ierr = IGACreateVec(iga,&ResS);CHKERRQ(ierr);
  ierr = VecZeroEntries(ResS);CHKERRQ(ierr);
  //

  Vec Res;
  ierr = IGACreateVec(iga,&Res);CHKERRQ(ierr);
  ierr = VecZeroEntries(Res);CHKERRQ(ierr);

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

  par->stepNumber  = user.StepRestart;
  par->currentTime = user.StepRestart*par->timeStep;
  ierr = input(par,manager,&user);CHKERRQ(ierr);
  manager.sync();
  if(par->stepNumber == 0){
  manager.connectVertsToTasks(false,&user);
  manager.sync();
 }

  if (par->stepNumber == 0){
    MPI_Allreduce(&user.totalInitialExplosiveVolume, &totalInitialExplosiveVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&par->numNodes, &totalNumNodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Read %d Immersed Particles from %d Foreground files\n", totalNumNodes, mpi_size);
    manager.connectVertsToTasks(false, &user);
    manager.sync();
  }

    //Initial condition or Read from Restarts
  if (user.StepRestart == 0){
    ierr = FormInitialCondition(iga,t,user.V0,&user);CHKERRQ(ierr);
  }
  else {
    ierr = ReadLastResults(par,user.V0,user.A0,user.StepRestart,user.FreqRestarts,manager,&user);CHKERRQ(ierr);
    manager.connectVertsToTasks(false,&user);
    manager.sync();
    MPI_Allreduce(&user.totalInitialExplosiveVolume, &totalInitialExplosiveVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&par->numNodes, &totalNumNodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    PetscPrintf(PETSC_COMM_WORLD, "Read %d Immersed Particles from %d Foreground files\n", totalNumNodes, mpi_size);
  }
  manager.sync();

     // Dump Initial Solution
     char filename[256];
     sprintf(filename,"velS%d.dat",par->stepNumber);
     ierr = IGAWriteVec(user.iga,user.V0,filename);CHKERRQ(ierr);

     Teuchos::RCP<Teuchos::ParameterList> peridigmParams(new Teuchos::ParameterList);
     RCP<ParameterList> discParams = rcp(new ParameterList);

     Teuchos::ParameterList& discretizationParams = peridigmParams->sublist("Discretization");
     discretizationParams.set("Type", "Text File");
     discretizationParams.set("Input Mesh File", "Plate1.txt");
     discretizationParams.set("InfluenceFunction", "Parabolic Decay");
     discretizationParams.set("Node Set", "Node_Set_1");
     discParams->set("Type", "Text File");
     discParams->set("Input Mesh File", "Plate1.txt");
     discParams->set("InfluenceFunction", "Parabolic Decay");
     discParams->set("Node Set", "Node_Set_1");

     // Apply displacement boundary conditions to immersed solid
     // Currrently, the edges are fully clamped (Air Backed Plate experimental setup)
     // Comment side if applying traction or using implicit traction free on a boundary
     // Nodeset generated by "NodeSetGenerator.m" file included in pre-processor. Need to call this function
     // with the name of the .txt file passed to discretizationParams above.
     Teuchos::ParameterList& boundaryConditions = peridigmParams->sublist("BoundaryConditions");
     boundaryConditions.set("Node Set Left", "Left.txt");
     boundaryConditions.set("Node Set Right", "Right.txt");
     boundaryConditions.set("Node Set Top", "Top.txt");
     boundaryConditions.set("Node Set Bottom", "Bottom.txt");

     // //Velocity
     // boundaryConditions.sublist("Prescribed Velocity Bottom X");
     // boundaryConditions.sublist("Prescribed Velocity Bottom X").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Bottom X").set("Node Set", "Node Set Bottom");
     // boundaryConditions.sublist("Prescribed Velocity Bottom X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Velocity Bottom X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Bottom Y");
     // boundaryConditions.sublist("Prescribed Velocity Bottom Y").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Bottom Y").set("Node Set", "Node Set Bottom");
     // boundaryConditions.sublist("Prescribed Velocity Bottom Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Velocity Bottom Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Velocity Bottom Z");
     boundaryConditions.sublist("Prescribed Velocity Bottom Z").set("Type", "Initial Velocity");
     boundaryConditions.sublist("Prescribed Velocity Bottom Z").set("Node Set", "Node Set Bottom");
     boundaryConditions.sublist("Prescribed Velocity Bottom Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Velocity Bottom Z").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Top X");
     // boundaryConditions.sublist("Prescribed Velocity Top X").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Top X").set("Node Set", "Node Set Top");
     // boundaryConditions.sublist("Prescribed Velocity Top X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Velocity Top X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Top Y");
     // boundaryConditions.sublist("Prescribed Velocity Top Y").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Top Y").set("Node Set", "Node Set Top");
     // boundaryConditions.sublist("Prescribed Velocity Top Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Velocity Top Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Velocity Top Z");
     boundaryConditions.sublist("Prescribed Velocity Top Z").set("Type", "Initial Velocity");
     boundaryConditions.sublist("Prescribed Velocity Top Z").set("Node Set", "Node Set Top");
     boundaryConditions.sublist("Prescribed Velocity Top Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Velocity Top Z").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Left X");
     // boundaryConditions.sublist("Prescribed Velocity Left X").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Left X").set("Node Set", "Node Set Left");
     // boundaryConditions.sublist("Prescribed Velocity Left X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Velocity Left X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Left Y");
     // boundaryConditions.sublist("Prescribed Velocity Left Y").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Left Y").set("Node Set", "Node Set Left");
     // boundaryConditions.sublist("Prescribed Velocity Left Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Velocity Left Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Velocity Left Z");
     boundaryConditions.sublist("Prescribed Velocity Left Z").set("Type", "Initial Velocity");
     boundaryConditions.sublist("Prescribed Velocity Left Z").set("Node Set", "Node Set Left");
     boundaryConditions.sublist("Prescribed Velocity Left Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Velocity Left Z").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Right X");
     // boundaryConditions.sublist("Prescribed Velocity Right X").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Right X").set("Node Set", "Node Set Right");
     // boundaryConditions.sublist("Prescribed Velocity Right X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Velocity Right X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Velocity Right Y");
     // boundaryConditions.sublist("Prescribed Velocity Right Y").set("Type", "Initial Velocity");
     // boundaryConditions.sublist("Prescribed Velocity Right Y").set("Node Set", "Node Set Right");
     // boundaryConditions.sublist("Prescribed Velocity Right Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Velocity Right Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Velocity Right Z");
     boundaryConditions.sublist("Prescribed Velocity Right Z").set("Type", "Initial Velocity");
     boundaryConditions.sublist("Prescribed Velocity Right Z").set("Node Set", "Node Set Right");
     boundaryConditions.sublist("Prescribed Velocity Right Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Velocity Right Z").set("Value", "0.0");

     // Displacement
     // boundaryConditions.sublist("Prescribed Displacement Bottom X");
     // boundaryConditions.sublist("Prescribed Displacement Bottom X").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Bottom X").set("Node Set", "Node Set Bottom");
     // boundaryConditions.sublist("Prescribed Displacement Bottom X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Displacement Bottom X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Bottom Y");
     // boundaryConditions.sublist("Prescribed Displacement Bottom Y").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Bottom Y").set("Node Set", "Node Set Bottom");
     // boundaryConditions.sublist("Prescribed Displacement Bottom Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Displacement Bottom Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Displacement Bottom Z");
     boundaryConditions.sublist("Prescribed Displacement Bottom Z").set("Type", "Prescribed Displacement");
     boundaryConditions.sublist("Prescribed Displacement Bottom Z").set("Node Set", "Node Set Bottom");
     boundaryConditions.sublist("Prescribed Displacement Bottom Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Displacement Bottom Z").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Top X");
     // boundaryConditions.sublist("Prescribed Displacement Top X").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Top X").set("Node Set", "Node Set Top");
     // boundaryConditions.sublist("Prescribed Displacement Top X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Displacement Top X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Top Y");
     // boundaryConditions.sublist("Prescribed Displacement Top Y").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Top Y").set("Node Set", "Node Set Top");
     // boundaryConditions.sublist("Prescribed Displacement Top Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Displacement Top Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Displacement Top Z");
     boundaryConditions.sublist("Prescribed Displacement Top Z").set("Type", "Prescribed Displacement");
     boundaryConditions.sublist("Prescribed Displacement Top Z").set("Node Set", "Node Set Top");
     boundaryConditions.sublist("Prescribed Displacement Top Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Displacement Top Z").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Left X");
     // boundaryConditions.sublist("Prescribed Displacement Left X").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Left X").set("Node Set", "Node Set Left");
     // boundaryConditions.sublist("Prescribed Displacement Left X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Displacement Left X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Left Y");
     // boundaryConditions.sublist("Prescribed Displacement Left Y").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Left Y").set("Node Set", "Node Set Left");
     // boundaryConditions.sublist("Prescribed Displacement Left Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Displacement Left Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Displacement Left Z");
     boundaryConditions.sublist("Prescribed Displacement Left Z").set("Type", "Prescribed Displacement");
     boundaryConditions.sublist("Prescribed Displacement Left Z").set("Node Set", "Node Set Left");
     boundaryConditions.sublist("Prescribed Displacement Left Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Displacement Left Z").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Right X");
     // boundaryConditions.sublist("Prescribed Displacement Right X").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Right X").set("Node Set", "Node Set Right");
     // boundaryConditions.sublist("Prescribed Displacement Right X").set("Coordinate", "x");
     // boundaryConditions.sublist("Prescribed Displacement Right X").set("Value", "0.0");

     // boundaryConditions.sublist("Prescribed Displacement Right Y");
     // boundaryConditions.sublist("Prescribed Displacement Right Y").set("Type", "Prescribed Displacement");
     // boundaryConditions.sublist("Prescribed Displacement Right Y").set("Node Set", "Node Set Right");
     // boundaryConditions.sublist("Prescribed Displacement Right Y").set("Coordinate", "y");
     // boundaryConditions.sublist("Prescribed Displacement Right Y").set("Value", "0.0");

     boundaryConditions.sublist("Prescribed Displacement Right Z");
     boundaryConditions.sublist("Prescribed Displacement Right Z").set("Type", "Prescribed Displacement");
     boundaryConditions.sublist("Prescribed Displacement Right Z").set("Node Set", "Node Set Right");
     boundaryConditions.sublist("Prescribed Displacement Right Z").set("Coordinate", "z");
     boundaryConditions.sublist("Prescribed Displacement Right Z").set("Value", "0.0");

  Teuchos::ParameterList& materialParams = peridigmParams->sublist("Materials");
  materialParams.sublist("My Shell Material");
  materialParams.sublist("My Shell Material").set("Material Model", "Elastic Shell");
  materialParams.sublist("My Shell Material").set("Density", par->density); //Kg/m^3
  materialParams.sublist("My Shell Material").set("Young's Modulus", par->youngModulus); //Pa
  materialParams.sublist("My Shell Material").set("Poisson's Ratio", par->poissonRatio);
  //materialParams.sublist("My Shell Material").set("G12", par->shearModulus);
  materialParams.sublist("My Shell Material").set("Shell Thickness", user.thickness); //m
  materialParams.sublist("My Shell Material").set("Number Of Shell Layers", 2);
  materialParams.sublist("My Shell Material").set("Expected Normal Direction X", "0.0");
  materialParams.sublist("My Shell Material").set("Expected Normal Direction Y", "0.0");
  materialParams.sublist("My Shell Material").set("Expected Normal Direction Z", "1.0");

  // These have to be consistent, otherwise there will be a conflict with disc and then peridigm obj.
  ParameterList blockParameterList;
  ParameterList& blockParams = blockParameterList.sublist("My Block");
  blockParams.set("Block Names", "block_1");
  blockParams.set("Material", "My Shell Material");
  blockParams.set("Horizon", user.horizon);

  Teuchos::ParameterList& BlockParams = peridigmParams->sublist("Blocks");
  BlockParams.sublist("My Block");
  BlockParams.sublist("My Block").set("Block Names", "block_1");
  BlockParams.sublist("My Block").set("Material", "My Shell Material");
  BlockParams.sublist("My Block").set("Horizon", user.horizon);
  //

  // Pass two solver params. One is a dummy parameter list and is never used and the other
  // is used to set up for restarts.
  user.PDInitialTime = round( user.PDInitialTime * 100000000 ) / 100000000.0;
  Teuchos::ParameterList& SolverParams = peridigmParams->sublist("Solver");
  SolverParams.sublist("Verlet");
  SolverParams.sublist("Verlet").set("Fixed dt", par->timeStep);
  SolverParams.set("Initial Time", user.PDInitialTime);
  SolverParams.set("Final Time", user.OutputRestart);
  SolverParams.set("Verbose", false);
  int num_solutions = (int)floor((user.OutputRestart-user.PDInitialTime)/(par->timeStep*par->FreqResults));

  RCP<ParameterList> solverParams = rcp(new ParameterList);
  solverParams->sublist("Verlet");
  solverParams->sublist("Verlet").set("Fixed dt", par->timeStep);
  solverParams->set("Initial Time", user.PDInitialTime);
  solverParams->set("Final Time", user.OutputRestart);
  solverParams->set("Verbose", false);
  //////////////////////////////////////////////////

  Teuchos::ParameterList& outputParams = peridigmParams->sublist("Output1");
  outputParams.set("Output File Type", "ExodusII");
  outputParams.set("NumProc", mpi_size);

  // temporary fix for exodus output after restart
  string exodusName = to_string(user.StepRestart) + "AirBackedPlate";

  //outputParams.set("Output Filename", "AirBackedPlate1");
  outputParams.set("Output Filename", exodusName);
  outputParams.set("Initial Output Step", 0);
  outputParams.set("Final Output Step", RestartStep);
  outputParams.set("Output Frequency", 1);
  outputParams.sublist("Output Variables");
  //outputParams.sublist("Output Variables").set("Volume", true);
  outputParams.sublist("Output Variables").set("Displacement", true);
  outputParams.sublist("Output Variables").set("Velocity", true);
  outputParams.sublist("Output Variables").set("Force", true);
  //outputParams.sublist("Output Variables").set("Velocity_Gradient", true);
  outputParams.sublist("Output Variables").set("Green_Lagrange_Strain", true);
  //outputParams.sublist("Output Variables").set("Left_Stretch_Tensor", true);
  //outputParams.sublist("Output Variables").set("Rotation_Tensor", true);
  outputParams.sublist("Output Variables").set("Cauchy_Stress", true);
  outputParams.sublist("Output Variables").set("Proc_Num", true);

  Teuchos::ParameterList& restartParams = peridigmParams->sublist("Restart");
  restartParams.set("Restart", PETSC_TRUE);
  bool restart = peridigmParams->isParameter("Restart");

  PetscPrintf(PETSC_COMM_WORLD, "\n#########################################################\n");
  PetscPrintf(PETSC_COMM_WORLD, "***Peridigm Restarts Active: %d ; Dumping Restarts at: %d***\n", restart, RestartStep);
  PetscPrintf(PETSC_COMM_WORLD, "***This run will need to progress at least %d Steps!***\n", RestartStep-par->stepNumber);
  PetscPrintf(PETSC_COMM_WORLD, "#########################################################\n\n");

  MPI_Barrier(MPI_COMM_WORLD);
  PeridigmNS::HorizonManager::self().loadHorizonInformationFromBlockParameters(blockParameterList);
  Teuchos::RCP<PeridigmNS::Discretization> textDiscretization(new PeridigmNS::TextFileDiscretization(epetraComm, discParams));
  Teuchos::RCP<PeridigmNS::Peridigm> peridigm(new PeridigmNS::Peridigm(PETSC_COMM_WORLD, peridigmParams, textDiscretization));
  PetscPrintf(PETSC_COMM_WORLD, "Done Initializing Discretization & PD Object\n\n");
  MPI_Barrier(MPI_COMM_WORLD);

  Teuchos::RCP<Epetra_Vector> currentPosition = peridigm->getY();
  int num_PD_nodes_onRank = currentPosition->MyLength()/3.0;
  ierr = MPI_Allreduce(&num_PD_nodes_onRank, &num_PD_nodes, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

  user.numFluidNodes =  totalNumNodes - num_PD_nodes;
  PetscPrintf(PETSC_COMM_WORLD, "%d PD nodes and %d immersed Fluid Nodes\n", num_PD_nodes, user.numFluidNodes);
  if(num_PD_nodes<=mpi_size){
    PetscPrintf(PETSC_COMM_WORLD, "Error: Too many processors for number of PD nodes! (%d nodes and %d cores)\n", num_PD_nodes, mpi_size);
    exit(0);
  }

  if(user.StepRestart==0){
  ierr = ParticleDistribute(par, &user, manager);CHKERRQ(ierr);
  manager.sync();
 }

  peridigm->setTimeStep(par->timeStep);

  if (par->stepNumber == 0){
  // Initialize tensors associated with all foreground particles; These quantities are not set inside
  // "Input"

    BGL_FORALL_VERTICES(v,manager.graph,Graph){
      FieldData fd = get(fd_property,v);
      ParticleInfo info = get(info_property,v);
      if(!info.isTask){

        fd.Inside = 1;
        fd.Boundary = 0;
        fd.damage = 0;
        fd.damage0 = 0;

      for (j=0;j<3;j++){
        fd.totalPhysicalAcceleration[j] = 0.0;
        fd.totalPhysicalDisplacement[j] = 0.0;
        fd.totalPhysicalVelocity[j] = 0.0;

        fd.totalPhysicalDisplacementOldStep[j] = 0.0;
        fd.totalPhysicalVelocityOldStep[j] = 0.0;
        fd.totalPhysicalAccelerationOldStep[j] = 0.0;

        fd.totalPhysicalDisplacementOldIteration[j] = 0.0;
        fd.totalPhysicalVelocityOldIteration[j] = 0.0;
        fd.totalPhysicalAccelerationOldIteration[j] = 0.0;

        fd.AccelerationIncrement[j] = 0.0;
        fd.interpolatedVelocity[j] = 0.0;

        fd.referenceCoord[j] = fd.currentCoord[j];

      }

      for(j = 0 ; j < 9 ; j++){
      fd.currentDeformationGradient[j] = 0.0;
      fd.alphaDeformationGradient[j] = 0.0;
      fd.DeformationGradientOld[j] = 0.0;

      fd.velocityGradient[j] = 0.0;
      }

      fd.DeformationGradientOld[0] = 1.0;
      fd.DeformationGradientOld[4] = 1.0;
      fd.DeformationGradientOld[8] = 1.0;

      fd.alphaDeformationGradient[0] = 1.0;
      fd.alphaDeformationGradient[4] = 1.0;
      fd.alphaDeformationGradient[8] = 1.0;

      fd.currentDeformationGradient[0] = 1.0;
      fd.currentDeformationGradient[4] = 1.0;
      fd.currentDeformationGradient[8] = 1.0;

      fd.determinantAlphaDeformationGradient = 1.0;
      fd.determinantCurrentDeformationGradient = 1.0;

      put(fd_property,v,fd);
    }
  }
}
  manager.sync();

  const PetscScalar *arrayU;
  const PetscScalar *arrayV;
  Vec               localV;
  Vec               localU;


  // Malloc needs to be outside of any loops
  ierr = PetscMalloc1(num_PD_nodes*3,&user.GlobalForces);CHKERRQ(ierr);
  ierr = PetscMalloc1(num_PD_nodes*3,&user.COORD);CHKERRQ(ierr);
  ierr = PetscMalloc1(num_PD_nodes*3,&user.GlobalVel);CHKERRQ(ierr);
  ierr = PetscMalloc1(num_PD_nodes*3,&user.GlobalDisp);CHKERRQ(ierr);
  ierr = PetscMalloc1(num_PD_nodes*3,&user.NORMAL);CHKERRQ(ierr);
  ierr = PetscMalloc1(num_PD_nodes,&user.VOLUME);CHKERRQ(ierr);

  // Need to declare as new otherwise the size of these arrays will exceed
  // the limit imposed by the stack frame
  double* VEL = new double[num_PD_nodes*3];
  double* DISP = new double[num_PD_nodes*3];
  double* FORCE = new double[num_PD_nodes*3];
  double* COORD = new double[num_PD_nodes*3];
  double* NORMAL = new double[num_PD_nodes*3];
  double* VOLUME = new double[num_PD_nodes];

  while(par->currentTime+par->timeStep <= par->finalTime+1e-9){
    PetscPrintf(PETSC_COMM_WORLD,"######################################################### \n");
    PetscPrintf(PETSC_COMM_WORLD,"Step Number: %d  Time step: %e, Time: %e \n",par->stepNumber, par->timeStep, par->currentTime);
    PetscPrintf(PETSC_COMM_WORLD,"######################################################### \n");

    par->currentTime+=par->timeStep;
    user.stepNumber  =par->stepNumber;

  if(par->stepNumber==0){
    peridigm->writePeridigmSubModel(0);
  }
    //Fluid predictor
    ierr = TSPredictStage_GeneralizedAlpha(&user,par);CHKERRQ(ierr);

   //Apply lagrangian update (newmark formulas) to foreground
   BGL_FORALL_VERTICES(v,manager.graph,Graph){
    FieldData fd = get(fd_property,v);
    ParticleInfo info = get(info_property,v);
     if(!info.isTask){
       if (fd.Boundary == 0){
         fd.totalPhysicalVelocityOldStep[0] = fd.totalPhysicalVelocity[0];
         fd.totalPhysicalVelocityOldStep[1] = fd.totalPhysicalVelocity[1];
         fd.totalPhysicalVelocityOldStep[2] = fd.totalPhysicalVelocity[2];
         fd.totalPhysicalAccelerationOldStep[0] = fd.totalPhysicalAcceleration[0];
         fd.totalPhysicalAccelerationOldStep[1] = fd.totalPhysicalAcceleration[1];
         fd.totalPhysicalAccelerationOldStep[2] = fd.totalPhysicalAcceleration[2];
         fd.totalPhysicalDisplacementOldStep[0] = fd.totalPhysicalDisplacement[0];
         fd.totalPhysicalDisplacementOldStep[1] = fd.totalPhysicalDisplacement[1];
         fd.totalPhysicalDisplacementOldStep[2] = fd.totalPhysicalDisplacement[2];

         fd.totalPhysicalAcceleration[0] = (user.Gamma - 1)/user.Gamma*fd.totalPhysicalAcceleration[0];
         fd.totalPhysicalAcceleration[1] = (user.Gamma - 1)/user.Gamma*fd.totalPhysicalAcceleration[1];
         fd.totalPhysicalAcceleration[2] = (user.Gamma - 1)/user.Gamma*fd.totalPhysicalAcceleration[2];

         fd.totalPhysicalDisplacement[0] = fd.totalPhysicalDisplacementOldStep[0] + par->timeStep*fd.totalPhysicalVelocityOldStep[0];
         fd.totalPhysicalDisplacement[0] += par->timeStep*par->timeStep/2*((1-2*user.Beta)*fd.totalPhysicalAccelerationOldStep[0]+2*user.Beta*fd.totalPhysicalAcceleration[0]);

         fd.totalPhysicalDisplacement[1] = fd.totalPhysicalDisplacementOldStep[1] + par->timeStep*fd.totalPhysicalVelocityOldStep[1];
         fd.totalPhysicalDisplacement[1] += par->timeStep*par->timeStep/2*((1-2*user.Beta)*fd.totalPhysicalAccelerationOldStep[1]+2*user.Beta*fd.totalPhysicalAcceleration[1]);

         fd.totalPhysicalDisplacement[2] = fd.totalPhysicalDisplacementOldStep[2] + par->timeStep*fd.totalPhysicalVelocityOldStep[2];
         fd.totalPhysicalDisplacement[2] += par->timeStep*par->timeStep/2*((1-2*user.Beta)*fd.totalPhysicalAccelerationOldStep[2]+2*user.Beta*fd.totalPhysicalAcceleration[2]);

       }else{
         fd.totalPhysicalVelocityOldStep[0] = 0.0;
         fd.totalPhysicalVelocityOldStep[1] = 0.0;
         fd.totalPhysicalVelocityOldStep[2] = 0.0;
         fd.totalPhysicalAccelerationOldStep[0] = 0.0;
         fd.totalPhysicalAccelerationOldStep[1] = 0.0;
         fd.totalPhysicalAccelerationOldStep[2] = 0.0;
         fd.totalPhysicalDisplacementOldStep[0] = 0.0;
         fd.totalPhysicalDisplacementOldStep[1] = 0.0;
         fd.totalPhysicalDisplacementOldStep[2] = 0.0;

         fd.totalPhysicalAcceleration[0] = 0.0;
         fd.totalPhysicalAcceleration[1] = 0.0;
         fd.totalPhysicalAcceleration[2] = 0.0;
         fd.totalPhysicalDisplacement[0] = 0.0;
         fd.totalPhysicalDisplacement[1] = 0.0;
         fd.totalPhysicalDisplacement[2] = 0.0;
       }

       if (fd.material == 0){
           fd.DeformationGradientOld[0] = fd.currentDeformationGradient[0];
           fd.DeformationGradientOld[1] = fd.currentDeformationGradient[1];
           fd.DeformationGradientOld[2] = fd.currentDeformationGradient[2];
           fd.DeformationGradientOld[3] = fd.currentDeformationGradient[3];
           fd.DeformationGradientOld[4] = fd.currentDeformationGradient[4];
           fd.DeformationGradientOld[5] = fd.currentDeformationGradient[5];
           fd.DeformationGradientOld[6] = fd.currentDeformationGradient[6];
           fd.DeformationGradientOld[7] = fd.currentDeformationGradient[7];
           fd.DeformationGradientOld[8] = fd.currentDeformationGradient[8];
       }
       put(fd_property,v,fd);
     }
   }

   PetscInt i,j,k,m,l, it;
   PetscInt dof = iga->dof;
   PetscInt dim = iga->dim;

   Mat A0;
   ierr = IGACreateMat(iga,&A0);CHKERRQ(ierr);

   BGL_FORALL_VERTICES(v,manager.graph,Graph){
   ParticleInfo info = get(info_property,v);
   FieldData fd = get(fd_property,v);
    if(!info.isTask){
      for(j=0;j<3;j++){
        info.tempCoord[j] = info.currentCoord[j];
        fd.computed_tempCoord[j] = fd.computed_currentCoord[j];
      }
      put(info_property,v,info);
      put(fd_property, v, fd);
    }
    }
   manager.sync();

    for (it=0; it<user.max_its ; it++) {
       PetscPrintf(PETSC_COMM_WORLD,":::::::::::::::::::::::::::::::::\n");
       PetscPrintf(PETSC_COMM_WORLD,"  Iteration: %d  \n", it);
       PetscPrintf(PETSC_COMM_WORLD,":::::::::::::::::::::::::::::::::\n");
       user.it = it;

       //////////////////////////////////////////
       Teuchos::RCP<Epetra_Vector> currentPosition = peridigm->getY();
       Teuchos::RCP<Epetra_Vector> displacement = peridigm->getU();
       Teuchos::RCP<Epetra_Vector> velocity = peridigm->getV();
       num_PD_nodes_onRank = (int)velocity->MyLength()/3.0;
       MPI_Barrier(MPI_COMM_WORLD);

       // Obtain Global node ID's (GID) for points on the current rank in order (numbered 0-N-1 instead of 1-N)!!
       // without this, cannot map from arrays with ID to Peridigm ID
       // i.e. force[0] = FORCE[GID[0]]
       // force[1] = FORCE[GID[0]+1]
       // force[2] = FORCE[GID[0]+2] ....
       int GID[num_PD_nodes_onRank];
       Teuchos::RCP< std::vector<PeridigmNS::Block> > blocks = peridigm->getBlocks();
       for(std::vector<Block>::iterator blockIt = blocks->begin() ; blockIt != blocks->end() ; blockIt++){
         std::string blockName = blockIt->getName();
         int numOwnedPoints = blockIt->getNeighborhoodData()->NumOwnedPoints();
         Teuchos::RCP<const Epetra_BlockMap> map = blockIt->getOwnedScalarPointMap();
         for(int i=0 ; i<numOwnedPoints ; i++){
           GID[i] = map->GID(i);
         }
       }
       MPI_Barrier(MPI_COMM_WORLD);
       ////////////////////////////////////////////////////////////////////////
       ierr = TSUpdateAlphaLevels_GeneralizedAlpha(&user);CHKERRQ(ierr);

       PetscReal t = par->currentTime;
       PetscReal dt = par->timeStep;
       PetscReal Alpha_m = user.Alpha_m;
       PetscReal Alpha_f = user.Alpha_f;
       par->Alpha_f = Alpha_f;
       PetscReal stage_time = t + Alpha_f*dt;

      ierr = IGAGetLocalVecArray(iga,user.Va,&localU,&arrayU);CHKERRQ(ierr);
      ierr = IGAGetLocalVecArray(iga,user.Aa,&localV,&arrayV);CHKERRQ(ierr);
      manager.sync();

       BGL_FORALL_VERTICES(v,manager.graph,Graph){
       FieldData fd = get(fd_property,v);
       ParticleInfo info = get(info_property,v);
        if(!info.isTask){
          fd.Inside = 0;
          put(fd_property,v,fd);
        }
       }
       manager.sync();
       BGL_FORALL_VERTICES(v,manager.graph,Graph){
       FieldData fd = get(fd_property,v);
       ParticleInfo info = get(info_property,v);
        if(!info.isTask){
         if (fd.Boundary == 0){
           fd.totalPhysicalVelocityOldIteration[0] = fd.totalPhysicalVelocity[0];
           fd.totalPhysicalVelocityOldIteration[1] = fd.totalPhysicalVelocity[1];
           fd.totalPhysicalVelocityOldIteration[2] = fd.totalPhysicalVelocity[2];
           fd.totalPhysicalAccelerationOldIteration[0] = fd.totalPhysicalAcceleration[0];
           fd.totalPhysicalAccelerationOldIteration[1] = fd.totalPhysicalAcceleration[1];
           fd.totalPhysicalAccelerationOldIteration[2] = fd.totalPhysicalAcceleration[2];
           fd.totalPhysicalDisplacementOldIteration[0] = fd.totalPhysicalDisplacement[0];
           fd.totalPhysicalDisplacementOldIteration[1] = fd.totalPhysicalDisplacement[1];
           fd.totalPhysicalDisplacementOldIteration[2] = fd.totalPhysicalDisplacement[2];

         fd.totalPhysicalVelocity[0] = fd.totalPhysicalVelocityOldStep[0] + Alpha_f*(fd.totalPhysicalVelocity[0] - fd.totalPhysicalVelocityOldStep[0]);
         fd.totalPhysicalDisplacement[0] = fd.totalPhysicalDisplacementOldStep[0] + Alpha_f*(fd.totalPhysicalDisplacement[0] - fd.totalPhysicalDisplacementOldStep[0]);
         fd.totalPhysicalVelocity[1] = fd.totalPhysicalVelocityOldStep[1] + Alpha_f*(fd.totalPhysicalVelocity[1] - fd.totalPhysicalVelocityOldStep[1]);
         fd.totalPhysicalDisplacement[1] = fd.totalPhysicalDisplacementOldStep[1] + Alpha_f*(fd.totalPhysicalDisplacement[1] - fd.totalPhysicalDisplacementOldStep[1]);
         fd.totalPhysicalVelocity[2] = fd.totalPhysicalVelocityOldStep[2] + Alpha_f*(fd.totalPhysicalVelocity[2] - fd.totalPhysicalVelocityOldStep[2]);
         fd.totalPhysicalDisplacement[2] = fd.totalPhysicalDisplacementOldStep[2] + Alpha_f*(fd.totalPhysicalDisplacement[2] - fd.totalPhysicalDisplacementOldStep[2]);
           if(fd.material==0){
             fd.totalPhysicalAcceleration[0] = fd.totalPhysicalAccelerationOldStep[0] + Alpha_m*(fd.totalPhysicalAcceleration[0] - fd.totalPhysicalAccelerationOldStep[0]);
             fd.totalPhysicalAcceleration[1] = fd.totalPhysicalAccelerationOldStep[1] + Alpha_m*(fd.totalPhysicalAcceleration[1] - fd.totalPhysicalAccelerationOldStep[1]);
             fd.totalPhysicalAcceleration[2] = fd.totalPhysicalAccelerationOldStep[2] + Alpha_m*(fd.totalPhysicalAcceleration[2] - fd.totalPhysicalAccelerationOldStep[2]);
           }
         }else{
           fd.totalPhysicalVelocityOldIteration[0] = 0.0;
           fd.totalPhysicalVelocityOldIteration[1] =0.0;
           fd.totalPhysicalVelocityOldIteration[2] =0.0;
           fd.totalPhysicalAccelerationOldIteration[0] = 0.0;
           fd.totalPhysicalAccelerationOldIteration[1] = 0.0;
           fd.totalPhysicalAccelerationOldIteration[2] = 0.0;
           fd.totalPhysicalDisplacementOldIteration[0] =0.0;
           fd.totalPhysicalDisplacementOldIteration[1] =0.0;
           fd.totalPhysicalDisplacementOldIteration[2] =0.0;

           fd.totalPhysicalVelocity[0] = 0.0;
           fd.totalPhysicalVelocity[1] =0.0;
           fd.totalPhysicalVelocity[2] =0.0;
           fd.totalPhysicalDisplacement[0] = 0.0;
           fd.totalPhysicalDisplacement[1] = 0.0;
           fd.totalPhysicalDisplacement[2] = 0.0;
         }
           for(j=0;j<3;j++){
             if(fd.material!=0){
             info.currentCoord[j] = info.tempCoord[j] + fd.totalPhysicalDisplacement[j] - fd.totalPhysicalDisplacementOldStep[j];
           }
           if(fd.material==0){
             fd.computed_currentCoord[j] = fd.computed_tempCoord[j] + fd.totalPhysicalDisplacement[j] - fd.totalPhysicalDisplacementOldStep[j];
            // info.currentCoord[0] = fd.computed_currentCoord[0];
            // info.currentCoord[1] = fd.computed_currentCoord[1];
           }
            }
          put(info_property,v,info);
          put(fd_property,v,fd);
        }
        }
       manager.sync();
       manager.connectVertsToTasks(true,&user);
       manager.sync();

       BGL_FORALL_VERTICES(v,manager.graph,Graph){
       FieldData fd = get(fd_property,v);
       ParticleInfo info = get(info_property,v);
      if(!info.isTask){
        if ((info.currentCoord[0] >= 0.0) && (info.currentCoord[0] <= user.Lx)){
        if ((info.currentCoord[1] >= 0.0) && (info.currentCoord[1] <= user.Ly)){
        if ((info.currentCoord[2] >= 0.0) && (info.currentCoord[2] <= user.Lz)){
        fd.Inside = 1;
        }}}
      }
        put(fd_property,v,fd);
      }
      manager.sync();

      //// Handoff Kinematic quantites (Velocity, displacement and current coordinate) (IGA->Peridigm) ////
      for(i = 0 ; i < num_PD_nodes*3 ; i++){
          user.GlobalVel[i]  = 0.0;
          user.GlobalDisp[i] = 0.0;
          user.COORD[i] = 0.0;
          VEL[i] = 0.0;
          DISP[i] = 0.0;
          COORD[i] = 0.0;
        }
      MPI_Barrier(PETSC_COMM_WORLD);

      BGL_FORALL_VERTICES(v,manager.graph,Graph){
        FieldData fd = get(fd_property,v);
        ParticleInfo info = get(info_property,v);
        if(!info.isTask && fd.material == 0 && fd.Boundary==0){
          if(fd.ID_PD<0){
            PetscPrintf(PETSC_COMM_WORLD, "ERROR! Negative PD_ID index!\n");
            exit(0);
          }
          for(j = 0 ; j < 3 ; j++){
          user.GlobalVel[fd.ID_PD*3+j] = fd.totalPhysicalVelocity[j];
          user.GlobalDisp[fd.ID_PD*3+j] = fd.totalPhysicalDisplacement[j];
          user.COORD[fd.ID_PD*3+j] = fd.computed_currentCoord[j];

          }
        }
        put(fd_property, v, fd);
      }
      manager.sync();
      MPI_Barrier(PETSC_COMM_WORLD);

      ierr = MPI_Allreduce(user.GlobalVel,VEL, num_PD_nodes*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
      ierr = MPI_Allreduce(user.GlobalDisp,DISP, num_PD_nodes*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
      ierr = MPI_Allreduce(user.COORD,COORD, num_PD_nodes*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

      for(i = 0 ; i < num_PD_nodes_onRank ; i++){
        for(j = 0 ; j < 3 ; j++){
          (*velocity)[i*3+j] = VEL[GID[i]*3+j];
          (*displacement)[i*3+j] = DISP[GID[i]*3+j];
          (*currentPosition)[i*3+j] = COORD[GID[i]*3+j];
        }
      }
      MPI_Barrier(PETSC_COMM_WORLD);
      ///////////////////////////////////////

       //// Volume Update for PD Particles ////
       ierr = updateVolumeAndDensity(par,&user,manager);CHKERRQ(ierr);
       manager.sync();
       //// Volume Update for Immersed-Fluid Particles ////
       ierr = ComputeCurrentExplosiveVolume(&user, par, manager);CHKERRQ(ierr);
       manager.sync();
      ////////////////////////////////////////

  totalExplosiveMass = 0.0;
  totalExplosiveVolume = 0.0;
  MPI_Allreduce(&user.totalExplosiveMass, &totalExplosiveMass, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  MPI_Allreduce(&user.totalCurrentExplosiveVolume, &totalExplosiveVolume, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  PetscPrintf(PETSC_COMM_WORLD, "Charge Mass = %e Charge Volume = %e Average Charge Density = %e\n", totalExplosiveMass, totalExplosiveVolume, totalExplosiveMass/totalExplosiveVolume);

  if(par->stepNumber == 0 && it == 0){
    ierr = outputTXT(par,manager);CHKERRQ(ierr);
  }

  //Compute Peridynamic force state:
  peridigm->computeInternalForce();

  // Get RCPs to important data fields ON THIS PROCESSOR

  Teuchos::RCP<Epetra_Vector> normal = peridigm->getNormal();
  Teuchos::RCP<Epetra_Vector> force = peridigm->getForce();
  Teuchos::RCP<Epetra_Vector> Volume = peridigm->getVolume();
  MPI_Barrier(PETSC_COMM_WORLD);

  ////// Force and Current Coordinate Handoff (Peridigm->IGA) /////
  for(i = 0; i < num_PD_nodes*3 ; i++){
    user.GlobalForces[i] = 0.0;
    user.NORMAL[i] = 0.0;
    FORCE[i] = 0.0;
    NORMAL[i] = 0.0;
  }
  for(i = 0 ; i < num_PD_nodes ; i++){
    VOLUME[i] = 0.0;
    user.VOLUME[i] = 0.0;
  }

  for(i = 0 ; i < num_PD_nodes_onRank; i++){
    for(j = 0 ; j < 3 ; j++){
      user.GlobalForces[GID[i]*3+j]   = (*force)[i*3+j];
      user.NORMAL[GID[i]*3+j]         = (*normal)[i*3+j];
      user.VOLUME[GID[i]]             = (*Volume)[i];
    }
  //  PetscPrintf(PETSC_COMM_SELF, "Force output=%e %e %e\n", (*force)[i*3+0], (*force)[i*3+1], (*force)[i*3+2]);
  }

  ierr = MPI_Allreduce(user.GlobalForces,FORCE,num_PD_nodes*3,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(user.NORMAL,NORMAL,num_PD_nodes*3,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(user.VOLUME,VOLUME,num_PD_nodes,MPI_DOUBLE,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);

  BGL_FORALL_VERTICES(v,manager.graph,Graph){
    FieldData fd = get(fd_property,v);
    ParticleInfo info = get(info_property,v);
    if(!info.isTask && fd.material == 0){

      if(fd.ID_PD<0){
        PetscPrintf(PETSC_COMM_WORLD, "ERROR! Negative PD_ID index! This means that particles were never distributed correctly!\n");
        exit(0);
      }
      for(j = 0 ; j < 3 ; j++){
        fd.internalForce[j]  = FORCE[fd.ID_PD*3+j]/fd.nodalVolume; //Convert force back to force density to compute strong solid residual
        fd.normal[j]         = NORMAL[fd.ID_PD*3+j];
        //fd.nodalVolume       = VOLUME[fd.ID_PD];
        fd.bodyForce[j]      = 0.0;
      }
      PetscReal norm = sqrt(fd.normal[0]*fd.normal[0]+fd.normal[1]*fd.normal[1]+fd.normal[2]*fd.normal[2]);
      if(norm>1e-7){
        for(j = 0 ; j < 3 ; j++){
          fd.normal[j] = fd.normal[j]/norm;
        }
      }else{
        PetscPrintf(PETSC_COMM_SELF, "Zero norm normal vector detected!\n");
        exit(0);
      }
    }
    put(fd_property, v, fd);
  }
  manager.sync();

////////////////////////////////////////////////////////////////////

  ////// Assembly of Solid Residual ///////////////////////////////
  // Compute PD-weak residual based on updated state //
  // After solving for the increment, update the quantities above
  // and computeInternalForce again //
  ierr = computeInertia(par,&user,manager);CHKERRQ(ierr);
  manager.sync();
  ierr = getInterpolatedVelocity(par, &user, manager);CHKERRQ(ierr);
  manager.sync();
  ierr = computePenaltyOnFluid(par, &user, ResS, manager);
  manager.sync();
  ///////////////////////////////////////////////////////////////

  ///// Mass Matricies /////
  ierr = IGAComputeMassFS(par, Alpha_m, user.Va, &user, MassFS, manager);CHKERRQ(ierr);
  manager.sync();
  MPI_Barrier(PETSC_COMM_WORLD);
  //////////////////////////

  ///// Integrated Tangent ///////
  ierr = IGAComputeIJacobianComp(iga,Alpha_m,user.Aa,stage_time,user.Va,A0,&user);CHKERRQ(ierr);
  MPI_Barrier(PETSC_COMM_WORLD);
  ///////////////////////////////

  ///// Add Mass Matricies /////////////
  ierr = MatAXPY(A0,-1.0,MassFS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  //////////////////////////////////////

  ///// Compute Fluid residual contributions F+(_-F_s) ////
  ierr = VecZeroEntries(ResF);CHKERRQ(ierr);
  ierr = IGAComputeIFunction(iga,dt,user.Aa,stage_time,user.Va,ResF);CHKERRQ(ierr);
  ierr = IGAComputeFS(iga, par, dt, stage_time, user.Aa, user.Va, &user, ResFS, manager);CHKERRQ(ierr);
  manager.sync();
  ///////////////////////////////////////////////////

  double ResNormSolid;
  VecNorm(ResS,NORM_2,&ResNormSolid);
  PetscPrintf(PETSC_COMM_WORLD,"Res. Norm Solid = %e\n", ResNormSolid);
  // Assemble Total Residual
  ierr = VecZeroEntries(Res);CHKERRQ(ierr);
  ierr = VecAYPX(Res,0.0,ResF);CHKERRQ(ierr);
  ierr = VecAYPX(Res,1.0,ResFS);CHKERRQ(ierr);
  ierr = VecAYPX(Res,1.0,ResS);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,user.Va,&localU,&arrayU);CHKERRQ(ierr);
  ierr = IGARestoreLocalVecArray(iga,user.Aa,&localV,&arrayV);CHKERRQ(ierr);
  //////////////////////////

  PetscScalar bc_rho[5]={0.0};
  bc_rho[0]=1.0;
  PetscScalar bc_ux[5]={0.0};
  bc_ux[1]=1.0;
  PetscScalar bc_uy[5]={0.0};
  bc_uy[2]=1.0;
  PetscScalar bc_uz[5]={0.0};
  bc_uz[3]=1.0;
  PetscScalar bc_temp[5]={0.0};
  bc_temp[4]=1.0;

  //// Apply Boundary Conditions through A0 ////
  PetscReal h_x = user.Lx/user.iga->elem_sizes[0];
  PetscReal h_f = user.horizon/3.0;
  PetscReal XX,YY,ZZ;
  PetscReal offset = 0.5*(0.305-0.5);
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
             PetscInt index_array[5]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             MatSetValuesLocal(A0,1,&index,5,index_array,bc_ux,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,5,index_array,1,&index,bc_ux,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }

           if((iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+1] <=0.00001+user.spacing/2.0) || (iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+1] >= user.Ly-0.00001-user.spacing/2.0)) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             PetscInt index_array[5]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             MatSetValuesLocal(A0,1,&index,5,index_array,bc_uy,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,5,index_array,1,&index,bc_uy,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }

           if(/*(iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+2] <=0.00001+user.spacing/2.0) ||*/ (iga->geometryX[(m*gnodesX*gnodesY + l*gnodesX+k)*dim+2] >= user.Lz-0.00001-user.spacing/2.0)) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             PetscInt index_array[5]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             MatSetValuesLocal(A0,1,&index,5,index_array,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,5,index_array,1,&index,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }
           // End boundary reflection

           // Boundary conditions on plane occupied initially by plate
            if(ZZ<0.000001){
            // U_z = 0 on all of the boundary
            // U_x*n <= 0
            // u_y*n <= 0 (should emerge naturally)
            //Clamped BC
            if(abs(XX)<=(0.305-0.254)/2.0 || abs(0.305-XX)<=(0.305-0.254)/2.0 || abs(YY)<=(0.305-0.254)/2.0 || abs(0.305-YY)<=(0.305-0.254)/2.0) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             PetscInt index_array[5]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             MatSetValuesLocal(A0,1,&index,5,index_array,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,5,index_array,1,&index,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }

           //Simply supported
           if(XX<=h_f*0.5 || XX >= 0.305-h_f*0.5 || YY<=h_f*0.5 ||  YY >= 0.305-h_f*0.5 ) {
            PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
            PetscInt index_array[5]={0};
            index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
            index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
            index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
            index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
            index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
            MatSetValuesLocal(A0,1,&index,5,index_array,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
            MatSetValuesLocal(A0,5,index_array,1,&index,bc_uz,INSERT_VALUES);CHKERRQ(ierr);
            VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
          }
        }

           //Isothermal condition (Necessary for water-JWL):
            if(PETSC_TRUE) {
             PetscInt index = (m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             PetscInt index_array[5]={0};
             index_array[0]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof;
             index_array[1]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+1;
             index_array[2]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+2;
             index_array[3]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+3;
             index_array[4]=(m*gnodesX*gnodesY + l*gnodesX+ k)*dof+4;
             MatSetValuesLocal(A0,1,&index,5,index_array,bc_temp,INSERT_VALUES);CHKERRQ(ierr);
             MatSetValuesLocal(A0,5,index_array,1,&index,bc_temp,INSERT_VALUES);CHKERRQ(ierr);
             VecSetValueLocal(Res,index,0.0,INSERT_VALUES);CHKERRQ(ierr);
           }
          }
      }
  }
  //// End Boundary conditions ////////////////////

  ierr = MatAssemblyBegin(A0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A0,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Res);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Res);CHKERRQ(ierr);
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

  ///// Solve to find increment ////////////
  KSP ksp;
  ierr = IGACreateKSP(iga,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A0,A0);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(ksp, PETSC_TRUE);
  ierr = KSPSolve(ksp,Res,user.dA);CHKERRQ(ierr);
  ierr = TSUpdateStage_GeneralizedAlpha(&user,par,user.dA);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  manager.sync();
  ////////////////////////////

  //// Apply Increment to background and interpolate on foreground //////
  const PetscScalar *arrayU1;
  Vec               localU1;
  ierr = IGAGetLocalVecArray(iga,user.V1,&localU1,&arrayU1);CHKERRQ(ierr);
  manager.sync();
  ierr = interpolateVelocityOnSolidNodes(par,&user, manager);CHKERRQ(ierr);
  manager.sync();
  ierr = IGARestoreLocalVecArray(iga,user.V1,&localU1,&arrayU1);CHKERRQ(ierr);
  manager.sync();
  /////////////////////////////////////////

  ierr = computeSolidResidualStrong(par, &user, manager);CHKERRQ(ierr);
  manager.sync();
  ierr = computePenaltyOnSolid(par,&user,manager);CHKERRQ(ierr);
  manager.sync();

  ///// Update kinematic quantites on forerground for next iteration //////
  BGL_FORALL_VERTICES(v,manager.graph,Graph){
                 FieldData fd = get(fd_property,v);
                 ParticleInfo info = get(info_property,v);
                  if(!info.isTask){
                   if ((fd.Boundary == 0) && (fd.Inside == 1) && (fd.material!=0)){

                       fd.AccelerationIncrement[0] = fd.totalPhysicalVelocity[0] - fd.totalPhysicalVelocityOldIteration[0];
                       fd.AccelerationIncrement[0] /= user.Gamma*par->timeStep;

                       fd.AccelerationIncrement[1] = fd.totalPhysicalVelocity[1] - fd.totalPhysicalVelocityOldIteration[1];
                       fd.AccelerationIncrement[1] /= user.Gamma*par->timeStep;

                       fd.AccelerationIncrement[2] = fd.totalPhysicalVelocity[2] - fd.totalPhysicalVelocityOldIteration[2];
                       fd.AccelerationIncrement[2] /= user.Gamma*par->timeStep;

                   }else{
                    if(fd.material==0){
                      // Solve independently on the solid domain for the acceleration
                      fd.AccelerationIncrement[0] = - fd.residual[0]/par->density;
                      fd.AccelerationIncrement[1] = - fd.residual[1]/par->density;
                      fd.AccelerationIncrement[2] = - fd.residual[2]/par->density;
                    }
                   }

                    fd.totalPhysicalAcceleration[0] = fd.totalPhysicalAccelerationOldIteration[0] + fd.AccelerationIncrement[0];
                    fd.totalPhysicalAcceleration[1] = fd.totalPhysicalAccelerationOldIteration[1] + fd.AccelerationIncrement[1];
                    fd.totalPhysicalAcceleration[2] = fd.totalPhysicalAccelerationOldIteration[2] + fd.AccelerationIncrement[2];

                    if(fd.material==0){
                      fd.totalPhysicalVelocity[0] = fd.totalPhysicalVelocityOldIteration[0] + user.Gamma*par->timeStep*fd.AccelerationIncrement[0];
                      fd.totalPhysicalVelocity[1] = fd.totalPhysicalVelocityOldIteration[1] + user.Gamma*par->timeStep*fd.AccelerationIncrement[1];
                      fd.totalPhysicalVelocity[2] = fd.totalPhysicalVelocityOldIteration[2] + user.Gamma*par->timeStep*fd.AccelerationIncrement[2];
                    }
                   if (fd.Boundary == 0){

                     fd.totalPhysicalDisplacement[0] = fd.totalPhysicalDisplacementOldIteration[0];
                     fd.totalPhysicalDisplacement[0] += user.Beta*par->timeStep*par->timeStep*fd.AccelerationIncrement[0];

                      fd.totalPhysicalDisplacement[1] = fd.totalPhysicalDisplacementOldIteration[1];
                      fd.totalPhysicalDisplacement[1] += user.Beta*par->timeStep*par->timeStep*fd.AccelerationIncrement[1];

                      fd.totalPhysicalDisplacement[2] = fd.totalPhysicalDisplacementOldIteration[2];
                      fd.totalPhysicalDisplacement[2] += user.Beta*par->timeStep*par->timeStep*fd.AccelerationIncrement[2];

                   }else{
                     fd.totalPhysicalDisplacement[0] =0.0;
                     fd.totalPhysicalDisplacement[1] =0.0;
                     fd.totalPhysicalDisplacement[2] =0.0;
                   }


                  put(fd_property,v,fd);

                  }
                  }
          manager.sync();

         BGL_FORALL_VERTICES(v,manager.graph,Graph){
         FieldData fd = get(fd_property,v);
         ParticleInfo info = get(info_property,v);
          if(!info.isTask){
                for(j=0;j<3;j++){
                  if(fd.material!=0){
                    info.currentCoord[j] = info.tempCoord[j] + fd.totalPhysicalDisplacement[j] - fd.totalPhysicalDisplacementOldStep[j] ;
                }
                if(fd.material==0){
                  fd.computed_currentCoord[j] = fd.computed_tempCoord[j] + fd.totalPhysicalDisplacement[j] - fd.totalPhysicalDisplacementOldStep[j] ;
                //  info.currentCoord[0] = fd.computed_currentCoord[0];
                //  info.currentCoord[1] = fd.computed_currentCoord[1];
                }
              }
            put(fd_property,v,fd);
            put(info_property,v,info);
          }
         }
         manager.sync();
         manager.connectVertsToTasks(true,&user);
         manager.sync();
/////////////////////////////////////////////

//// Update the Displacement, current position, velocity of PD particles to update the force-State ////
//// These arrays are structured as:  ////
//// velx_ID vely_ID velz_ID velx_ID+1 ....  ////
//// To keep track of which particles on this processor correspond to particles in the Peridigm code, use  ////
//// a map between the ID from preprocessor to the place in the array as:  ////
//// i=0 -> ID_PD=0=ID-num_fluid_nodes-1, read from the preprocessor. ////

//// Handoff Kinematic quantites (Velocity, displacement and current coordinate) (IGA->Peridigm) ////
for(i = 0 ; i < num_PD_nodes*3 ; i++){
    user.GlobalVel[i]  = 0.0;
    user.GlobalDisp[i] = 0.0;
    user.COORD[i] = 0.0;
    VEL[i] = 0.0;
    DISP[i] = 0.0;
    COORD[i] = 0.0;
  }
MPI_Barrier(PETSC_COMM_WORLD);

BGL_FORALL_VERTICES(v,manager.graph,Graph){
  FieldData fd = get(fd_property,v);
  ParticleInfo info = get(info_property,v);
  if(!info.isTask && fd.material == 0 && fd.Boundary==0){
    if(fd.ID_PD<0){
      PetscPrintf(PETSC_COMM_WORLD, "ERROR! Negative PD_ID index!\n");
      exit(0);
    }
    for(j = 0 ; j < 3 ; j++){
    user.GlobalVel[fd.ID_PD*3+j] = fd.totalPhysicalVelocity[j];
    user.GlobalDisp[fd.ID_PD*3+j] = fd.totalPhysicalDisplacement[j];
    user.COORD[fd.ID_PD*3+j] = fd.computed_currentCoord[j];
    }
  }
  put(fd_property, v, fd);
}
MPI_Barrier(PETSC_COMM_WORLD);
manager.sync();

ierr = MPI_Allreduce(user.GlobalVel,VEL, num_PD_nodes*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
ierr = MPI_Allreduce(user.GlobalDisp,DISP, num_PD_nodes*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);
ierr = MPI_Allreduce(user.COORD,COORD, num_PD_nodes*3, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);CHKERRQ(ierr);

for(i = 0 ; i < num_PD_nodes_onRank ; i++){
  for(j = 0 ; j < 3 ; j++){
  (*velocity)[i*3+j] = VEL[GID[i]*3+j];
  (*displacement)[i*3+j] = DISP[GID[i]*3+j];
  (*currentPosition)[i*3+j] = COORD[GID[i]*3+j];
  }
}
MPI_Barrier(PETSC_COMM_WORLD);
///////////////////////////////////////

}//###### END OF ITERATION LOOP ##########

//////////////////////////////////////////////////////////////////////////////////////////////////////

  ierr = VecCopy(user.V1,user.V0);CHKERRQ(ierr);
  ierr = VecCopy(user.A1,user.A0);CHKERRQ(ierr);
  ierr = VecCopy(user.D1,user.D0);CHKERRQ(ierr);

  //Update Solution State
  par->stepNumber++;
  peridigm->updateState();

   if (par->stepNumber % par->FreqResults == 0) {
        char filename[256];
        sprintf(filename,"velS%d.dat",par->stepNumber);
        ierr = IGAWriteVec(user.iga,user.V1,filename);CHKERRQ(ierr);
        //Write result to an ExodusII file at FreqResult Interval:
        peridigm->writePeridigmSubModel(par->stepNumber/par->FreqResults);
    }

    // Dump residual At restart interval
    if (par->stepNumber %  RestartStep == 0) {
      double t1, t2;
      MPI_Barrier(PETSC_COMM_WORLD);
      t1 = MPI_Wtime();
        peridigm->writeRestart(solverParams);
      MPI_Barrier(PETSC_COMM_WORLD);
      t2 = MPI_Wtime();
      PetscPrintf(PETSC_COMM_WORLD, "Restart Writing Time = %e\n", t2-t1);
        char filename[256];
        sprintf(filename,"ResS%d.dat",par->stepNumber);
        ierr = IGAWriteVec(user.iga,Res,filename);CHKERRQ(ierr);
    }

        // Dump Restart vector
        if (par->stepNumber % RestartStep == 0) {
          char filename[256];
          sprintf(filename,"velS%d.dat",par->stepNumber);
          ierr = IGAWriteVec(user.iga,user.V1,filename);CHKERRQ(ierr);
          sprintf(filename,"acelS%d.dat",par->stepNumber);
          ierr = IGAWriteVec(user.iga,user.A1,filename);CHKERRQ(ierr);
        }
        if (par->stepNumber % RestartStep == 0){
          ierr = OutputRestarts(par,user.V1,user.A1,manager);CHKERRQ(ierr);
        }
        if (par->stepNumber % RestartStep == RestartStep-1){
          ierr = OutputOldGeometry(par,manager);CHKERRQ(ierr);
        }

        ierr = MatDestroy(&A0);CHKERRQ(ierr);
        ierr = outputTXT(par,manager);CHKERRQ(ierr);
MPI_Barrier(PETSC_COMM_WORLD);
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
