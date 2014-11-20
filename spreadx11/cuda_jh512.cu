#include <stdint.h>

typedef struct {
    uint32_t x[8][4];                     /*the 1024-bit state, ( x[i][0] || x[i][1] || x[i][2] || x[i][3] ) is the ith row of the state in the pseudocode*/
    uint32_t buffer[16];                  /*the 512-bit message block to be hashed;*/
} hashState;

static uint32_t *d_nonce[8];
__constant__ uint32_t pTarget[8];

/*42 round constants, each round constant is 32-byte (256-bit)*/
__constant__ uint32_t c_INIT_bitslice[8][4];
__constant__ uint32_t c_E8_bitslice_roundconstant[42][8];

const uint32_t h_INIT_bitslice[8][4] = {
	{ 0x964bd16f, 0x17aa003e, 0x052e6a63, 0x43d5157a},
	{ 0x8d5e228a, 0x0bef970c, 0x591234e9, 0x61c3b3f2},
	{ 0xc1a01d89, 0x1e806f53, 0x6b05a92a, 0x806d2bea},
	{ 0xdbcc8e58, 0xa6ba7520, 0x763a0fa9, 0xf73bf8ba},
	{ 0x05e66901, 0x694ae341, 0x8e8ab546, 0x5ae66f2e},
	{ 0xd0a74710, 0x243c84c1, 0xb1716e3b, 0x99c15a2d},
	{ 0xecf657cf, 0x56f8b19d, 0x7c8806a7, 0x56b11657},
	{ 0xdffcc2e3, 0xfb1785e6, 0x78465a54, 0x4bdd8ccc} };

const uint32_t h_E8_bitslice_roundconstant[42][8] = {
    { 0xa2ded572, 0x67f815df, 0x0a15847b, 0x571523b7, 0x90d6ab81, 0xf6875a4d, 0xc54f9f4e, 0x402bd1c3 },
	{ 0xe03a98ea, 0x9cfa455c, 0x99d2c503, 0x9a99b266, 0xb4960266, 0x8a53bbf2, 0x1a1456b5, 0x31a2db88 },
	{ 0x5c5aa303, 0xdb0e199a, 0x0ab23f40, 0x1044c187, 0x8019051c, 0x1d959e84, 0xadeb336f, 0xdccde75e },
	{ 0x9213ba10, 0x416bbf02, 0x156578dc, 0xd027bbf7, 0x39812c0a, 0x5078aa37, 0xd2bf1a3f, 0xd3910041 },
	{ 0x0d5a2d42, 0x907eccf6, 0x9c9f62dd, 0xce97c092, 0x0ba75c18, 0xac442bc7, 0xd665dfd1, 0x23fcc663 },
	{ 0x036c6e97, 0x1ab8e09e, 0x7e450521, 0xa8ec6c44, 0xbb03f1ee, 0xfa618e5d, 0xb29796fd, 0x97818394 },
	{ 0x37858e4a, 0x2f3003db, 0x2d8d672a, 0x956a9ffb, 0x8173fe8a, 0x6c69b8f8, 0x4672c78a, 0x14427fc0 },
	{ 0x8f15f4c5, 0xc45ec7bd, 0xa76f4475, 0x80bb118f, 0xb775de52, 0xbc88e4ae, 0x1e00b882, 0xf4a3a698 },
	{ 0x338ff48e, 0x1563a3a9, 0x24565faa, 0x89f9b7d5, 0x20edf1b6, 0xfde05a7c, 0x5ae9ca36, 0x362c4206 },
	{ 0x433529ce, 0x3d98fe4e, 0x74f93a53, 0xa74b9a73, 0x591ff5d0, 0x86814e6f, 0x81ad9d0e, 0x9f5ad8af },
	{ 0x670605a7, 0x6a6234ee, 0xbe280b8b, 0x2717b96e, 0x26077447, 0x3f1080c6, 0x6f7ea0e0, 0x7b487ec6 },
	{ 0xa50a550d, 0xc0a4f84a, 0x9fe7e391, 0x9ef18e97, 0x81727686, 0xd48d6050, 0x415a9e7e, 0x62b0e5f3 },
	{ 0xec1f9ffc, 0x7a205440, 0x001ae4e3, 0x84c9f4ce, 0xf594d74f, 0xd895fa9d, 0x117e2e55, 0xa554c324 },
	{ 0x2872df5b, 0x286efebd, 0xe27ff578, 0xb2c4a50f, 0xef7c8905, 0x2ed349ee, 0x85937e44, 0x7f5928eb },
	{ 0x37695f70, 0x4a3124b3, 0xf128865e, 0x65e4d61d, 0x04771bc7, 0xe720b951, 0xe843fe74, 0x8a87d423 },
	{ 0xa3e8297d, 0xf2947692, 0x097acbdd, 0xc1d9309b, 0xfb301b1d, 0xe01bdc5b, 0x4f4924da, 0xbf829cf2 },
	{ 0x31bae7a4, 0xffbf70b4, 0x0544320d, 0x48bcf8de, 0x32fcae3b, 0x39d3bb53, 0xc1c39f45, 0xa08b29e0 },
	{ 0xfd05c9e5, 0x0f09aef7, 0x12347094, 0x34f19042, 0x01b771a2, 0x95ed44e3, 0x368e3be9, 0x4a982f4f },
	{ 0x631d4088, 0x15f66ca0, 0x4b44c147, 0xffaf5287, 0xf14abb7e, 0x30c60ae2, 0xc5b67046, 0xe68c6ecc },
	{ 0x56a4d5a4, 0x00ca4fbd, 0x4b849dda, 0xae183ec8, 0x45ce5773, 0xadd16430, 0x68cea6e8, 0x67255c14 },
	{ 0xf28cdaa3, 0x16e10ecb, 0x5806e933, 0x9a99949a, 0x20b2601f, 0x7b846fc2, 0x7facced1, 0x1885d1a0 },
	{ 0xa15b5932, 0xd319dd8d, 0xc01c9a50, 0x46b4a5aa, 0x67633d9f, 0xba6b04e4, 0xab19caf6, 0x7eee560b },
	{ 0xea79b11f, 0x742128a9, 0x35f7bde9, 0xee51363b, 0x5aac571d, 0x76d35075, 0xfec2463a, 0x01707da3 },
	{ 0xafc135f7, 0x42d8a498, 0x20eced78, 0x79676b9e, 0x15638341, 0xa8db3aea, 0x4d3bc3fa, 0x832c8332 },
	{ 0x1f3b40a7, 0xf347271c, 0x34f04059, 0x9a762db7, 0x6c4e3ee7, 0xfd4f21d2, 0x398dfdb8, 0xef5957dc },
	{ 0x490c9b8d, 0xdaeb492b, 0x49d7a25b, 0x0d70f368, 0xd0ae3b7d, 0x84558d7a, 0xf0e9a5f5, 0x658ef8e4 },
	{ 0xf4a2b8a0, 0x533b1036, 0x9e07a80c, 0x5aec3e75, 0x92946891, 0x4f88e856, 0x555cb05b, 0x4cbcbaf8 },
	{ 0x993bbbe3, 0x7b9487f3, 0xd6f4da75, 0x5d1c6b72, 0x28acae64, 0x6db334dc, 0x50a5346c, 0x71db28b8 },
	{ 0xf2e261f8, 0x2a518d10, 0x3364dbe3, 0xfc75dd59, 0xf1bcac1c, 0xa23fce43, 0x3cd1bb67, 0xb043e802 },
	{ 0xca5b0a33, 0x75a12988, 0x4d19347f, 0x5c5316b4, 0xc3943b92, 0x1e4d790e, 0xd7757479, 0x3fafeeb6 },
	{ 0xf7d4a8ea, 0x21391abe, 0x097ef45c, 0x5127234c, 0x5324a326, 0xd23c32ba, 0x4a17a344, 0xadd5a66d },
	{ 0xa63e1db5, 0x08c9f2af, 0x983d5983, 0x563c6b91, 0xa17cf84c, 0x4d608672, 0xcc3ee246, 0xf6c76e08 },
	{ 0xb333982f, 0x5e76bcb1, 0xa566d62b, 0x2ae6c4ef, 0xe8b6f406, 0x36d4c1be, 0x1582ee74, 0x6321efbc },
	{ 0x0d4ec1fd, 0x69c953f4, 0xc45a7da7, 0x26585806, 0x1614c17e, 0x16fae006, 0x3daf907e, 0x3f9d6328 },
	{ 0xe3f2c9d2, 0x0cd29b00, 0x30ceaa5f, 0x300cd4b7, 0x16512a74, 0x9832e0f2, 0xd830eb0d, 0x9af8cee3 },
	{ 0x7b9ec54b, 0x9279f1b5, 0x6ee651ff, 0xd3688604, 0x574d239b, 0x316796e6, 0xf3a6e6cc, 0x05750a17 },
	{ 0xd98176b1, 0xce6c3213, 0x8452173c, 0x62a205f8, 0xb3cb2bf4, 0x47154778, 0x825446ff, 0x486a9323 },
	{ 0x0758df38, 0x65655e4e, 0x897cfcf2, 0x8e5086fc, 0x442e7031, 0x86ca0bd0, 0xa20940f0, 0x4e477830 },
	{ 0x39eea065, 0x8338f7d1, 0x37e95ef7, 0xbd3a2ce4, 0x26b29721, 0x6ff81301, 0xd1ed44a3, 0xe7de9fef },
	{ 0x15dfa08b, 0xd9922576, 0xf6f7853c, 0xbe42dc12, 0x7ceca7d8, 0x7eb027ab, 0xda7d8d53, 0xdea83eaa },
	{ 0x93ce25aa, 0xd86902bd, 0xfd43f65a, 0xf908731a, 0xdaef5fc0, 0xa5194a17, 0x33664d97, 0x6a21fd4c },
	{ 0x3198b435, 0x701541db, 0xbb0f1eea, 0x9b54cded, 0xa163d09a, 0x72409751, 0xbf9d75f6, 0xe26f4791 }
};

/*swapping bit 2i with bit 2i+1 of 32-bit x*/
#define SWAP1(x)   (x) = ((((x) & 0x55555555UL) << 1) | (((x) & 0xaaaaaaaaUL) >> 1));
/*swapping bits 4i||4i+1 with bits 4i+2||4i+3 of 32-bit x*/
#define SWAP2(x)   (x) = ((((x) & 0x33333333UL) << 2) | (((x) & 0xccccccccUL) >> 2));
/*swapping bits 8i||8i+1||8i+2||8i+3 with bits 8i+4||8i+5||8i+6||8i+7 of 32-bit x*/
#define SWAP4(x)   (x) = ((((x) & 0x0f0f0f0fUL) << 4) | (((x) & 0xf0f0f0f0UL) >> 4));
/*swapping bits 16i||16i+1||......||16i+7  with bits 16i+8||16i+9||......||16i+15 of 32-bit x*/
//#define SWAP8(x)   (x) = ((((x) & 0x00ff00ffUL) << 8) | (((x) & 0xff00ff00UL) >> 8));
#define SWAP8(x) (x) = __byte_perm(x, x, 0x2301);
/*swapping bits 32i||32i+1||......||32i+15 with bits 32i+16||32i+17||......||32i+31 of 32-bit x*/
//#define SWAP16(x)  (x) = ((((x) & 0x0000ffffUL) << 16) | (((x) & 0xffff0000UL) >> 16));
#define SWAP16(x) (x) = __byte_perm(x, x, 0x1032);

/*The MDS transform*/
#define L(m0,m1,m2,m3,m4,m5,m6,m7) \
      (m6) ^= (m0) ^ (m3);         \
      (m5) ^= (m2);                \
      (m4) ^= (m1);                \
      (m7) ^= (m0);                \
      (m1) ^= (m6);                \
      (m3) ^= (m4);                \
      (m0) ^= (m5); \
      (m2) ^= (m4) ^ (m7); 

/*The Sbox*/
#define Sbox(m0,m1,m2,m3,cc)       \
      m3  = ~(m3);                 \
      m0 ^= ((~(m2)) & (cc));      \
      temp0 = (cc) ^ ((m0) & (m1));\
      m0 ^= ((m2) & (m3));         \
      m3 ^= ((~(m1)) & (m2));      \
      m1 ^= ((m0) & (m2));         \
      m2 ^= ((m0) & (~(m3)));      \
      m0 ^= ((m1) | (m3));         \
      m3 ^= ((m1) & (m2));         \
      m1 ^= (temp0 & (m0));        \
      m2 ^= temp0;

__device__ __forceinline__ void Sbox_and_MDS_layer(hashState* state, uint32_t roundnumber)
{
    uint32_t temp0;
    //Sbox and MDS layer
#pragma unroll 4
    for (int i = 0; i < 4; i++) {
        Sbox(state->x[0][i],state->x[2][i], state->x[4][i], state->x[6][i], c_E8_bitslice_roundconstant[roundnumber][i]);
        Sbox(state->x[1][i],state->x[3][i], state->x[5][i], state->x[7][i], c_E8_bitslice_roundconstant[roundnumber][i+4]);
        L(state->x[0][i],state->x[2][i],state->x[4][i],state->x[6][i],state->x[1][i],state->x[3][i],state->x[5][i],state->x[7][i]);
    }
}

__device__ __forceinline__ void RoundFunction0(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP1(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction1(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP2(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction2(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP4(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction3(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP8(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction4(hashState* state, uint32_t roundnumber)
{
	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 4
		for (int i = 0; i < 4; i++) SWAP16(state->x[j][i]);
	}
}

__device__ __forceinline__ void RoundFunction5(hashState* state, uint32_t roundnumber)
{
	uint32_t temp0;

	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 2
		for (int i = 0; i < 4; i = i+2) {
			temp0 = state->x[j][i]; state->x[j][i] = state->x[j][i+1]; state->x[j][i+1] = temp0;
		}
	}
}

__device__ __forceinline__ void RoundFunction6(hashState* state, uint32_t roundnumber)
{
	uint32_t temp0;

	Sbox_and_MDS_layer(state, roundnumber);

#pragma unroll 4
	for (int j = 1; j < 8; j = j+2)
	{
#pragma unroll 2
		for (int i = 0; i < 2; i++) {
			temp0 = state->x[j][i]; state->x[j][i] = state->x[j][i+2]; state->x[j][i+2] = temp0;
		}
	}
}

/*The bijective function E8, in bitslice form */
__device__ __forceinline__ void E8(hashState *state)
{
    /*perform 6 rounds*/
//#pragma unroll 6
    for (int i = 0; i < 42; i+=7)
	{
		RoundFunction0(state, i);
		RoundFunction1(state, i+1);
		RoundFunction2(state, i+2);
		RoundFunction3(state, i+3);
		RoundFunction4(state, i+4);
		RoundFunction5(state, i+5);
		RoundFunction6(state, i+6);
	}
}

/*The compression function F8 */
__device__ __forceinline__ void F8(hashState *state)
{
    /*xor the 512-bit message with the fist half of the 1024-bit hash state*/
#pragma unroll 16
    for (int i = 0; i < 16; i++)  state->x[i >> 2][i & 3] ^= ((uint32_t*)state->buffer)[i];

    /*the bijective function E8 */
    E8(state);

    /*xor the 512-bit message with the second half of the 1024-bit hash state*/
#pragma unroll 16
    for (int i = 0; i < 16; i++)  state->x[(16+i) >> 2][(16+i) & 3] ^= ((uint32_t*)state->buffer)[i];
}


__device__ __forceinline__ void JHHash(const uint32_t *data, uint32_t *hashval)
{
    hashState state;

#pragma unroll 8
	for(int j=0;j<8;j++)
	{
#pragma unroll 4
		for(int i=0;i<4;i++)
			state.x[j][i] = c_INIT_bitslice[j][i];
	}

#pragma unroll 16
    for (int i=0; i < 16; ++i) state.buffer[i] = data[i];
    F8(&state);

    /*pad the message when databitlen is multiple of 512 bits, then process the padded block*/
    state.buffer[0] = 0x80;
#pragma unroll 14
    for (int i=1; i < 15; i++) state.buffer[i] = 0;
    state.buffer[15] = 0x00020000;
    F8(&state);

    /*truncating the final hash value to generate the message digest*/
#pragma unroll 16
    for (int i=0; i < 16; ++i) hashval[i] = state.x[4][i];
}

__device__ __forceinline__ void JHHashC( int threads, int thread, uint32_t *hash )
{
    hashState state;

#pragma unroll 8
	for(int j=0;j<8;j++)
	{
#pragma unroll 4
		for(int i=0;i<4;i++)
			state.x[j][i] = c_INIT_bitslice[j][i];
	}

#pragma unroll 16
    for (int i=0; i < 16; ++i) state.buffer[i] = hash[i*threads+thread];
    F8(&state);

    /*pad the message when databitlen is multiple of 512 bits, then process the padded block*/
    state.buffer[0] = 0x80;
#pragma unroll 14
    for (int i=1; i < 15; i++) state.buffer[i] = 0;
    state.buffer[15] = 0x00020000;
    F8(&state);

    /*truncating the final hash value to generate the message digest*/
#pragma unroll 16
    for (int i=0; i < 16; ++i) hash[i*threads+thread] = state.x[4][i];
}

__device__ __forceinline__ void JHHash_final(const uint32_t *data, uint32_t *hashval)
{
    hashState state;

#pragma unroll 8
	for(int j=0;j<8;j++)
	{
#pragma unroll 4
		for(int i=0;i<4;i++)
			state.x[j][i] = c_INIT_bitslice[j][i];
	}

#pragma unroll 16
    for (int i=0; i < 16; ++i) state.buffer[i] = data[i];
    F8(&state);

    /*pad the message when databitlen is multiple of 512 bits, then process the padded block*/
    state.buffer[0] = 0x80;
#pragma unroll 14
    for (int i=1; i < 15; i++) state.buffer[i] = 0;
    state.buffer[15] = 0x00020000;
    F8(&state);

    /*truncating the final hash value to generate the message digest*/
#pragma unroll 8
    for (int i=0; i < 8; ++i) hashval[i] = state.x[4][i];
}

// Die Hash-Funktion
__global__ void quark_jh512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];

        JHHash(Hash, Hash);
    }
}

__global__ void quark_jh512c_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;

        JHHashC(threads, thread, (uint32_t *)g_hash);
    }
}

__global__ void quark_jh512_gpu_hash_final_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector, uint32_t *d_nonce)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *Hash = (uint32_t*)&g_hash[8 * hashPosition];
        uint32_t hash[8];

        JHHash_final(Hash, hash);
    
		int position = -1;
		bool rc = true;

#pragma unroll 8
		for (int i = 7; i >= 0; i--) {
			if (hash[i] > pTarget[i]) {
				if(position < i) {
					position = i;
					rc = false;
				}
	 		}
	 		if (hash[i] < pTarget[i]) {
				if(position < i) {
					position = i;
					rc = true;
				}
	 		}
		}

		if(rc == true)
				d_nonce[0] = nounce;
    }
}


// Setup-Funktionen
__host__ void quark_jh512_cpu_init(int thr_id, int threads)
{
	cudaMalloc(&d_nonce[thr_id], sizeof(uint32_t));
	
    cudaMemcpyToSymbol( c_E8_bitslice_roundconstant,
                        h_E8_bitslice_roundconstant,
                        sizeof(h_E8_bitslice_roundconstant),
                        0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol( c_INIT_bitslice,
                        h_INIT_bitslice,
                        sizeof(h_INIT_bitslice),
                        0, cudaMemcpyHostToDevice);
}

__host__ void quark_jh512_cpu_setTarget(const void *ptarget)
{
	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( pTarget, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_jh512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	cudaDeviceSynchronize();
}

__host__ void quark_jh512c_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 256;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_jh512c_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	cudaDeviceSynchronize();
}

__host__ uint32_t quark_jh512_cpu_hash_final_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 256;

    cudaMemset(d_nonce[thr_id], 0xffffffff, sizeof(uint32_t));

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_jh512_gpu_hash_final_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector, d_nonce[thr_id]);

	cudaDeviceSynchronize();

    uint32_t res;
    cudaMemcpy(&res, d_nonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return res;
}

