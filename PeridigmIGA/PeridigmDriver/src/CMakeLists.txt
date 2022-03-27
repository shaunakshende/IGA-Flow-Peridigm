
# Glob header files
file (GLOB srcHEADERS 
  "*.h*"
)

set(PeridigmDriver_HEADERS
    ${srcHEADERS}
)

# Glob source files
file (GLOB srcSOURCES 
  "*.c*"
)

set(PeridigmDriver_SOURCES
    ${srcSOURCES}
)

set(PeridigmDriver_LINK_LIBRARIES
${Required_LIBS}
)

add_executable(PeridigmDriver PeridigmDriver.cpp)
target_link_libraries(PeridigmDriver ${PeridigmDriver_LINK_LIBRARIES})
