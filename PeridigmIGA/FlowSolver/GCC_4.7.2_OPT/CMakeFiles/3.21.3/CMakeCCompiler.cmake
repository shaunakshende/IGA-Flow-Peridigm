set(CMAKE_C_COMPILER "/opt/apps/intel18/impi/18.0.2/bin/mpicc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Intel")
set(CMAKE_C_COMPILER_VERSION "18.0.2.20180210")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "11")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")
set(CMAKE_C17_COMPILE_FEATURES "")
set(CMAKE_C23_COMPILE_FEATURES "")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "GNU")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_C_SIMULATE_VERSION "6.3.0")




set(CMAKE_AR "/opt/apps/gcc/6.3.0/bin/ar")
set(CMAKE_C_COMPILER_AR "")
set(CMAKE_RANLIB "/opt/apps/gcc/6.3.0/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "")
set(CMAKE_LINKER "/opt/apps/xalt/xalt/bin/ld")
set(CMAKE_MT "")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_C_COMPILER_ENV_VAR "CC")

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/include;/opt/intel/compilers_and_libraries_2018.2.199/linux/pstl/include;/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/include;/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/include;/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/include;/opt/intel/compilers_and_libraries_2018.2.199/linux/ipp/include;/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/include/intel64;/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/include/icc;/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/include;/usr/local/include;/opt/apps/gcc/6.3.0/lib/gcc/x86_64-pc-linux-gnu/6.3.0/include;/opt/apps/gcc/6.3.0/lib/gcc/x86_64-pc-linux-gnu/6.3.0/include-fixed;/opt/apps/gcc/6.3.0/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "mpifort;mpi;mpigi;dl;rt;pthread;imf;svml;irng;m;ipgo;decimal;cilkrts;stdc++;gcc;gcc_s;irc;svml;c;gcc;gcc_s;irc_s;dl;c")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/lib/release_mt;/opt/intel/compilers_and_libraries_2018.2.199/linux/mpi/intel64/lib;/work2/07855/sshende/stampede2/boost_1_65_1/stage/lib;/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64_lin/gcc4.4;/opt/intel/compilers_and_libraries_2018.2.199/linux/daal/lib/intel64_lin;/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7;/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/lib/intel64_lin;/opt/intel/compilers_and_libraries_2018.2.199/linux/compiler/lib/intel64_lin;/opt/intel/compilers_and_libraries_2018.2.199/linux/ipp/lib/intel64;/opt/apps/gcc/6.3.0/lib/gcc/x86_64-pc-linux-gnu/6.3.0;/opt/apps/gcc/6.3.0/lib64;/lib64;/usr/lib64;/opt/apps/gcc/6.3.0/x86_64-pc-linux-gnu/lib;/opt/apps/gcc/6.3.0/lib;/lib;/usr/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
