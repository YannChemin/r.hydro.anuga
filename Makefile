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

# OpenCL program source, embedded as an array of lines: the shared
# definitions, the kernel bodies and the kernel wrappers, in that order.
OCL_SOURCES = cl/anuga_common.h cl/anuga_sw.h cl/anuga_kernels.cl
EXTRA_CLEAN_FILES = ocl_kernels_src.h

include $(MODULE_TOPDIR)/include/Make/Module.make

ocl_kernels_src.h: $(OCL_SOURCES)
	{ echo '/* Generated from $(OCL_SOURCES) by the Makefile. */'; \
	  echo 'static const char *ocl_kernel_lines[] = {'; \
	  cat $(OCL_SOURCES) | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' \
	      -e 's/^/    "/' -e 's/$$/\\n",/'; \
	  echo '};'; } > $@

$(OBJDIR)/kernels_ocl.o: ocl_kernels_src.h

default: cmd
