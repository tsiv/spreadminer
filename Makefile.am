AUTOMAKE_OPTIONS = subdir-objects

if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST		= autogen.sh README.txt LICENSE.txt \
			  spreadminer.sln spreadminer.vcxproj spreadminer.vcxproj.filters \
			  compat/gettimeofday.c compat/getopt/getopt_long.c cpuminer-config.h.in

SUBDIRS		= compat

bin_PROGRAMS	= spreadminer

spreadminer_SOURCES	= elist.h miner.h compat.h \
			  compat/inttypes.h compat/stdbool.h compat/unistd.h \
			  compat/sys/time.h compat/getopt/getopt.h \
			  cpu-miner.c util.c sph/bmw.c sph/blake.c sph/groestl.c sph/jh.c sph/keccak.c sph/skein.c scrypt.c sha2.c \
			  sph/cubehash.c sph/echo.c sph/luffa.c sph/shavite.c sph/simd.c \
			  sph/bmw.h sph/sph_blake.h sph/sph_groestl.h sph/sph_jh.h sph/sph_keccak.h sph/sph_skein.h sph/sph_types.h \
			  cuda_util.cu \
			  spreadx11/spreadx11.cu spreadx11/cuda_sha256.cu spreadx11/cuda_spreadx11_sign.cu \
			  spreadx11/cuda_blake.cu spreadx11/cuda_jh512.cu spreadx11/cuda_quark_groestl512.cu spreadx11/cuda_skein512.cu \
			  spreadx11/cuda_bmw512.cu spreadx11/cuda_quark_keccak512.cu \
			  spreadx11/cuda_x11_luffa512.cu spreadx11/cuda_x11_cubehash512.cu \
			  spreadx11/cuda_x11_shavite512.cu spreadx11/cuda_x11_simd512.cu spreadx11/cuda_x11_echo.cu

spreadminer_LDFLAGS	= $(PTHREAD_FLAGS) @CUDA_LDFLAGS@
spreadminer_LDADD	= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ @CUDA_LIBS@ @OPENMP_CFLAGS@ @LIBS@
spreadminer_CPPFLAGS	= -msse2 @LIBCURL_CPPFLAGS@ @OPENMP_CFLAGS@ $(PTHREAD_FLAGS) -fno-strict-aliasing $(JANSSON_INCLUDES) -DSCRYPT_KECCAK512 -DSCRYPT_CHACHA -DSCRYPT_CHOOSE_COMPILETIME
NVCC_GENCODE		= -gencode=arch=compute_50,code=\"sm_50,compute_50\"

# we're now targeting all major compute architectures within one binary.
.cu.o:
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" $(NVCC_GENCODE) --maxrregcount=80 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<

spreadx11/cuda_x11_shavite512.o: spreadx11/cuda_x11_shavite512.cu
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" $(NVCC_GENCODE) --maxrregcount=128 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<        

spreadx11/cuda_bmw512.o: spreadx11/cuda_bmw512.cu
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" -arch=compute_35 --maxrregcount=128 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<        

spreadx11/cuda_sha256.o: spreadx11/cuda_sha256.cu
	$(NVCC) @CFLAGS@ -I . -Xptxas "-abi=no -v" $(NVCC_GENCODE) --maxrregcount=64 --ptxas-options=-v $(JANSSON_INCLUDES) -o $@ -c $<   

