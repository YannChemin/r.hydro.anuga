MODULE_TOPDIR = ../..

PGM = r.hydro.anuga

LIBES = $(RASTERLIB) $(GISLIB) $(MATHLIB)
DEPENDENCIES = $(RASTERDEP) $(GISDEP)

# -ffp-contract=off: no fused multiply-add, so host geometry and the bed
# dequantisation are bitwise reproducible (PLAN.md section 4.7).
EXTRA_CFLAGS = $(OPENMP_CFLAGS) -std=c11 -ffp-contract=off \
               -Wall -Wextra -Wno-unused-parameter
EXTRA_INC = -I.
EXTRA_LIBS = -lOpenCL $(OPENMP_LIBPATH) $(OPENMP_LIB)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd
