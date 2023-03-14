rm -f CMakeCache.txt
rm -rf CMakeFiles
source ~/.bashrc
EXTRA_ARGS=$@
module load cmake

$WORK2/apps/cmake/bin/cmake \
-D CMAKE_BUILD_TYPE:STRING=Debug \
-D TRILINOS_DIR:PATH=/work2/07855/sshende/stampede2/projects/trilinos \
-D Peridigm_INCLUDE_DIR:PATH=/work2/07855/sshende/stampede2/peridigm/release/src/Include \
-D Peridigm_LIB_DIR:PATH=/work2/07855/sshende/stampede2/peridigm/release/src/lib \
-D PETSC_DIR:PATH=$PETSC_DIR \
-D PETSC_INCLUDE_DIRS:PATH=$PETSC_DIR/include \
-D PETIGA_DIR:PATH=/home1/07855/sshende/petsc-3.15.2/PetIGA \
-D CMAKE_C_COMPILER:STRING=`which mpicc` \
-D CMAKE_CXX_COMPILER:STRING=`which mpicxx` \
-D CMAKE_CXX_FLAGS:STRING="-mkl -O2 -march=icelake -mtune=icelake -Wall -ansi -pedantic -Wno-long-long -Wno-deprecated -std=c++14 -I ${TACC_EIGEN_INC}" \
-D CMAKE_C_FLAGS:STRING="-mkl -O2 -march=icelake -mtune=icelake -Wall -ansi -pedantic -Wno-long-long -Wno-deprecated" \
$EXTRA_ARGS \
..
