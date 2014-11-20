#include <stdint.h>
#include <stdio.h>
#include <memory.h>


#define ROTL_1(d0, d1, v0, v1)      ROTL_SMALL(d0, d1, v0, v1,  1)
#define ROTL_2(d0, d1, v0, v1)      ROTL_SMALL(d0, d1, v0, v1,  2)
#define ROTL_3(d0, d1, v0, v1)      ROTL_SMALL(d0, d1, v0, v1,  3)
#define ROTL_6(d0, d1, v0, v1)      ROTL_SMALL(d0, d1, v0, v1,  6)
#define ROTL_8(d0, d1, v0, v1)      ROTL_SMALL(d0, d1, v0, v1,  8)
#define ROTL_10(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 10)
#define ROTL_14(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 14)
#define ROTL_15(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 15)
#define ROTL_18(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 18)
#define ROTL_20(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 20)
#define ROTL_21(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 21)
#define ROTL_25(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 25)
#define ROTL_27(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 27)
#define ROTL_28(d0, d1, v0, v1)     ROTL_SMALL(d0, d1, v0, v1, 28)
#define ROTL_32(d0, d1, v0, v1)     (d0 = v1; d1 = v0; )
#define ROTL_36(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 36)
#define ROTL_39(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 39)
#define ROTL_41(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 41)
#define ROTL_43(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 43)
#define ROTL_44(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 44)
#define ROTL_45(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 45)
#define ROTL_55(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 55)
#define ROTL_56(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 56)
#define ROTL_61(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 61)
#define ROTL_62(d0, d1, v0, v1)     ROTL_BIG(d0, d1, v0, v1, 62)

#define ROTLI_1(d1, d2, v1, v2)    ROTLI_odd1(d1, d2, v1, v2)
#define ROTLI_2(d1, d2, v1, v2)    ROTLI_even(d1, d2, v1, v2,  1)
#define ROTLI_3(d1, d2, v1, v2)    ROTLI_odd( d1, d2, v1, v2,  2)
#define ROTLI_6(d1, d2, v1, v2)    ROTLI_even(d1, d2, v1, v2,  3)
#define ROTLI_8(d1, d2, v1, v2)    ROTLI_even(d1, d2, v1, v2,  4)
#define ROTLI_10(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2,  5)
#define ROTLI_14(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2,  7)
#define ROTLI_15(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2,  8)
#define ROTLI_18(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2,  9)
#define ROTLI_20(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2, 10)
#define ROTLI_21(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 11)
#define ROTLI_25(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 13)
#define ROTLI_27(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 14)
#define ROTLI_28(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2, 14)
#define ROTLI_36(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2, 18)
#define ROTLI_39(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 20)
#define ROTLI_41(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 21)
#define ROTLI_43(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 22)
#define ROTLI_44(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2, 22)
#define ROTLI_45(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 23)
#define ROTLI_55(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 28)
#define ROTLI_56(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2, 28)
#define ROTLI_61(d1, d2, v1, v2)   ROTLI_odd( d1, d2, v1, v2, 31)
#define ROTLI_62(d1, d2, v1, v2)   ROTLI_even(d1, d2, v1, v2, 31)

#define ROTs(a, b, n) ROTL_##n(s[a], s[a+1], s[b], s[b+1])
#define ROTIs(a, b, n) ROTLI_##n(s[a], s[a+1], s[b], s[b+1])

static __device__ __forceinline__ void ROTL_SMALL( uint32_t &d0, uint32_t &d1, uint32_t v0, uint32_t v1, const uint32_t offset )
{
#if __CUDA_ARCH__ >= 320
    asm(
        "shf.l.wrap.b32 %0, %2, %3, %4;\n\t"
        "shf.l.wrap.b32 %1, %3, %2, %4;\n\t"
        : "=r"(d0), "=r"(d1) 
        : "r"(v1), "r"(v0), "r"(offset));
#else
    d0 = (v0 << offset) | (v1 >> (32-offset));
    d1 = (v1 << offset) | (v0 >> (32-offset));
#endif
}

static __device__ __forceinline__ void ROTL_BIG( uint32_t &d0, uint32_t &d1, uint32_t v0, uint32_t v1, const uint32_t offset )
{
#if __CUDA_ARCH__ >= 320
    asm(
        "shf.l.wrap.b32 %0, %3, %2, %4;\n\t"
        "shf.l.wrap.b32 %1, %2, %3, %4;\n\t"
        : "=r"(d0), "=r"(d1) 
        : "r"(v1), "r"(v0), "r"(offset-32));
#else
    d0 = (v1 << (offset-32)) | (v0 >> (64-offset));
    d1 = (v0 << (offset-32)) | (v1 >> (64-offset));
#endif
}

static __device__ __forceinline__ void ROTLI_even( uint32_t &d1, uint32_t &d2, uint32_t v1, uint32_t v2, const int offset )
{
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(d1) : "r"(v1), "r"(v1), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(d2) : "r"(v2), "r"(v2), "r"(offset));
}

static __device__ __forceinline__ void ROTLI_odd( uint32_t &d1, uint32_t &d2, uint32_t v1, uint32_t v2, const int offset )
{
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(d1) : "r"(v2), "r"(v2), "r"(offset));
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(d2) : "r"(v1), "r"(v1), "r"(offset-1));
}

static __device__ __forceinline__ void ROTLI_odd1( uint32_t &d1, uint32_t &d2, uint32_t v1, uint32_t v2 )
{
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(d1) : "r"(v2), "r"(v2), "r"(1));
    d2 = v1;
}

static __device__ __forceinline__ void interleave( uint32_t &l, uint32_t &h )
{ 
    uint32_t t; 

    t = (l ^ (l >> 1)) & 0x22222222; l ^= t ^ (t << 1); 
    t = (h ^ (h >> 1)) & 0x22222222; h ^= t ^ (t << 1); 
    t = (l ^ (l >> 2)) & 0x0C0C0C0C; l ^= t ^ (t << 2); 
    t = (h ^ (h >> 2)) & 0x0C0C0C0C; h ^= t ^ (t << 2); 
    t = (l ^ (l >> 4)) & 0x00F000F0; l ^= t ^ (t << 4); 
    t = (h ^ (h >> 4)) & 0x00F000F0; h ^= t ^ (t << 4); 
    t = (l ^ (l >> 8)) & 0x0000FF00; l ^= t ^ (t << 8); 
    t = (h ^ (h >> 8)) & 0x0000FF00; h ^= t ^ (t << 8); 
    t = (l ^ (h << 16)) & 0xFFFF0000; 
    l ^= t; h ^= t >> 16; 
}

static __device__ __forceinline__ void deinterleave( uint32_t &l, uint32_t &h )
{
    uint32_t t; 

    t = (l ^ (h << 16)) & 0xFFFF0000; 
    l ^= t; h ^= t >> 16; 
    t = (l ^ (l >> 8)) & 0x0000FF00; l ^= t ^ (t << 8); 
    t = (h ^ (h >> 8)) & 0x0000FF00; h ^= t ^ (t << 8); 
    t = (l ^ (l >> 4)) & 0x00F000F0; l ^= t ^ (t << 4); 
    t = (h ^ (h >> 4)) & 0x00F000F0; h ^= t ^ (t << 4); 
    t = (l ^ (l >> 2)) & 0x0C0C0C0C; l ^= t ^ (t << 2); 
    t = (h ^ (h >> 2)) & 0x0C0C0C0C; h ^= t ^ (t << 2); 
    t = (l ^ (l >> 1)) & 0x22222222; l ^= t ^ (t << 1); 
    t = (h ^ (h >> 1)) & 0x22222222; h ^= t ^ (t << 1); 
}

__constant__ uint32_t d_RC[48];

static const uint32_t h_interleaved_RC[48] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000089,
    0x00000000, 0x8000008b, 0x00000000, 0x80008080,
    0x00000001, 0x0000008b, 0x00000001, 0x00008000,
    0x00000001, 0x80008088, 0x00000001, 0x80000082,
    0x00000000, 0x0000000b, 0x00000000, 0x0000000a,
    0x00000001, 0x00008082, 0x00000000, 0x00008003,
    0x00000001, 0x0000808b, 0x00000001, 0x8000000b,
    0x00000001, 0x8000008a, 0x00000001, 0x80000081,
    0x00000000, 0x80000081, 0x00000000, 0x80000008,
    0x00000000, 0x00000083, 0x00000000, 0x80008003,
    0x00000001, 0x80008088, 0x00000000, 0x80000088,
    0x00000001, 0x00008000, 0x00000000, 0x80008082
};

static const uint32_t h_RC[48] = {
    0x00000001, 0x00000000, 0x00008082, 0x00000000,
    0x0000808a, 0x80000000, 0x80008000, 0x80000000,
    0x0000808b, 0x00000000, 0x80000001, 0x00000000,
    0x80008081, 0x80000000, 0x00008009, 0x80000000,
    0x0000008a, 0x00000000, 0x00000088, 0x00000000,
    0x80008009, 0x00000000, 0x8000000a, 0x00000000,
    0x8000808b, 0x00000000, 0x0000008b, 0x80000000,
    0x00008089, 0x80000000, 0x00008003, 0x80000000,
    0x00008002, 0x80000000, 0x00000080, 0x80000000,
    0x0000800a, 0x00000000, 0x8000000a, 0x80000000,
    0x80008081, 0x80000000, 0x00008080, 0x80000000,
    0x80000001, 0x00000000, 0x80008008, 0x80000000
};

static uint32_t *d_nonce[8];
__constant__ uint32_t pTarget[8];

static __device__ void keccak_block(uint32_t *s) 
{
    uint32_t t[10], u[10], v[2];

#pragma unroll 4
    for (int i = 0; i < 48; i += 2) {

        t[4] = s[4] ^ s[14] ^ s[24] ^ s[34] ^ s[44];
        t[5] = s[5] ^ s[15] ^ s[25] ^ s[35] ^ s[45];
        t[2] = s[2] ^ s[12] ^ s[22] ^ s[32] ^ s[42];
        t[3] = s[3] ^ s[13] ^ s[23] ^ s[33] ^ s[43];
        t[6] = s[6] ^ s[16] ^ s[26] ^ s[36] ^ s[46];
        t[7] = s[7] ^ s[17] ^ s[27] ^ s[37] ^ s[47];
        t[8] = s[8] ^ s[18] ^ s[28] ^ s[38] ^ s[48];
        t[9] = s[9] ^ s[19] ^ s[29] ^ s[39] ^ s[49];
        t[0] = s[0] ^ s[10] ^ s[20] ^ s[30] ^ s[40];
        t[1] = s[1] ^ s[11] ^ s[21] ^ s[31] ^ s[41];
    
        ROTL_1(u[2], u[3], t[4], t[5]);
        ROTL_1(u[0], u[1], t[2], t[3]);
        ROTL_1(u[4], u[5], t[6], t[7]);
        ROTL_1(u[6], u[7], t[8], t[9]);
        ROTL_1(u[8], u[9], t[0], t[1]);
        
        u[2] ^= t[0]; u[3] ^= t[1];
        u[0] ^= t[8]; u[1] ^= t[9];
        u[4] ^= t[2]; u[5] ^= t[3];
        u[6] ^= t[4]; u[7] ^= t[5];
        u[8] ^= t[6]; u[9] ^= t[7];

        s[2] ^= u[2]; s[3] ^= u[3];

        s[0] ^= u[0]; s[10] ^= u[0]; s[20] ^= u[0]; s[30] ^= u[0]; s[40] ^= u[0];
        s[1] ^= u[1]; s[11] ^= u[1]; s[21] ^= u[1]; s[31] ^= u[1]; s[41] ^= u[1];
        s[12] ^= u[2]; s[22] ^= u[2]; s[32] ^= u[2]; s[42] ^= u[2];
        s[13] ^= u[3]; s[23] ^= u[3]; s[33] ^= u[3]; s[43] ^= u[3];
        s[4] ^= u[4]; s[14] ^= u[4]; s[24] ^= u[4]; s[34] ^= u[4]; s[44] ^= u[4];
        s[5] ^= u[5]; s[15] ^= u[5]; s[25] ^= u[5]; s[35] ^= u[5]; s[45] ^= u[5];
        s[6] ^= u[6]; s[16] ^= u[6]; s[26] ^= u[6]; s[36] ^= u[6]; s[46] ^= u[6];
        s[7] ^= u[7]; s[17] ^= u[7]; s[27] ^= u[7]; s[37] ^= u[7]; s[47] ^= u[7];
        s[8] ^= u[8]; s[18] ^= u[8]; s[28] ^= u[8]; s[38] ^= u[8]; s[48] ^= u[8];
        s[9] ^= u[9]; s[19] ^= u[9]; s[29] ^= u[9]; s[39] ^= u[9]; s[49] ^= u[9];

        v[0] = s[2]; v[1] = s[3];
        ROTs( 2, 12, 44);
        ROTs(12, 18, 20);
        ROTs(18, 44, 61);
        ROTs(44, 28, 39);
        ROTs(28, 40, 18);
        ROTs(40,  4, 62);
        ROTs( 4, 24, 43);
        ROTs(24, 26, 25);
        ROTs(26, 38,  8);
        ROTs(38, 46, 56);
        ROTs(46, 30, 41);
        ROTs(30,  8, 27);
        ROTs( 8, 48, 14);
        ROTs(48, 42,  2);
        ROTs(42, 16, 55);
        ROTs(16, 32, 45);
        ROTs(32, 10, 36);
        ROTs(10,  6, 28);
        ROTs( 6, 36, 21);
        ROTs(36, 34, 15);
        ROTs(34, 22, 10);
        ROTs(22, 14,  6);
        ROTs(14, 20,  3);
        ROTL_1(s[20], s[21], v[0], v[1]);

        v[0] = s[ 0]; v[1] = s[ 2]; s[ 0] ^= (~v[1]) & s[ 4]; s[ 2] ^= (~s[ 4]) & s[ 6]; s[ 4] ^= (~s[ 6]) & s[ 8]; s[ 6] ^= (~s[ 8]) & v[0]; s[ 8] ^= (~v[0]) & v[1];
        v[0] = s[ 1]; v[1] = s[ 3]; s[ 1] ^= (~v[1]) & s[ 5]; s[ 3] ^= (~s[ 5]) & s[ 7]; s[ 5] ^= (~s[ 7]) & s[ 9]; s[ 7] ^= (~s[ 9]) & v[0]; s[ 9] ^= (~v[0]) & v[1];
        v[0] = s[10]; v[1] = s[12]; s[10] ^= (~v[1]) & s[14]; s[12] ^= (~s[14]) & s[16]; s[14] ^= (~s[16]) & s[18]; s[16] ^= (~s[18]) & v[0]; s[18] ^= (~v[0]) & v[1];
        v[0] = s[11]; v[1] = s[13]; s[11] ^= (~v[1]) & s[15]; s[13] ^= (~s[15]) & s[17]; s[15] ^= (~s[17]) & s[19]; s[17] ^= (~s[19]) & v[0]; s[19] ^= (~v[0]) & v[1];
        v[0] = s[20]; v[1] = s[22]; s[20] ^= (~v[1]) & s[24]; s[22] ^= (~s[24]) & s[26]; s[24] ^= (~s[26]) & s[28]; s[26] ^= (~s[28]) & v[0]; s[28] ^= (~v[0]) & v[1];
        v[0] = s[21]; v[1] = s[23]; s[21] ^= (~v[1]) & s[25]; s[23] ^= (~s[25]) & s[27]; s[25] ^= (~s[27]) & s[29]; s[27] ^= (~s[29]) & v[0]; s[29] ^= (~v[0]) & v[1];
        v[0] = s[30]; v[1] = s[32]; s[30] ^= (~v[1]) & s[34]; s[32] ^= (~s[34]) & s[36]; s[34] ^= (~s[36]) & s[38]; s[36] ^= (~s[38]) & v[0]; s[38] ^= (~v[0]) & v[1];
        v[0] = s[31]; v[1] = s[33]; s[31] ^= (~v[1]) & s[35]; s[33] ^= (~s[35]) & s[37]; s[35] ^= (~s[37]) & s[39]; s[37] ^= (~s[39]) & v[0]; s[39] ^= (~v[0]) & v[1];
        v[0] = s[40]; v[1] = s[42]; s[40] ^= (~v[1]) & s[44]; s[42] ^= (~s[44]) & s[46]; s[44] ^= (~s[46]) & s[48]; s[46] ^= (~s[48]) & v[0]; s[48] ^= (~v[0]) & v[1];
        v[0] = s[41]; v[1] = s[43]; s[41] ^= (~v[1]) & s[45]; s[43] ^= (~s[45]) & s[47]; s[45] ^= (~s[47]) & s[49]; s[47] ^= (~s[49]) & v[0]; s[49] ^= (~v[0]) & v[1];

        s[0] ^= d_RC[i];
        s[1] ^= d_RC[i+1];
    }
}

static __device__ __forceinline__ void keccak_interleaved_block(uint32_t *s)
{
    uint32_t t[10], u[10], v[2];

#pragma unroll 4
    for (int i = 0; i < 48; i += 2) {

        t[4] = s[4] ^ s[14] ^ s[24] ^ s[34] ^ s[44];
        t[5] = s[5] ^ s[15] ^ s[25] ^ s[35] ^ s[45];
        t[2] = s[2] ^ s[12] ^ s[22] ^ s[32] ^ s[42];
        t[3] = s[3] ^ s[13] ^ s[23] ^ s[33] ^ s[43];
        t[6] = s[6] ^ s[16] ^ s[26] ^ s[36] ^ s[46];
        t[7] = s[7] ^ s[17] ^ s[27] ^ s[37] ^ s[47];
        t[8] = s[8] ^ s[18] ^ s[28] ^ s[38] ^ s[48];
        t[9] = s[9] ^ s[19] ^ s[29] ^ s[39] ^ s[49];
        t[0] = s[0] ^ s[10] ^ s[20] ^ s[30] ^ s[40];
        t[1] = s[1] ^ s[11] ^ s[21] ^ s[31] ^ s[41];
    
        ROTLI_1(u[2], u[3], t[4], t[5]);
        ROTLI_1(u[0], u[1], t[2], t[3]);
        ROTLI_1(u[4], u[5], t[6], t[7]);
        ROTLI_1(u[6], u[7], t[8], t[9]);
        ROTLI_1(u[8], u[9], t[0], t[1]);
        
        u[2] ^= t[0]; u[3] ^= t[1];
        u[0] ^= t[8]; u[1] ^= t[9];
        u[4] ^= t[2]; u[5] ^= t[3];
        u[6] ^= t[4]; u[7] ^= t[5];
        u[8] ^= t[6]; u[9] ^= t[7];

        s[2] ^= u[2]; s[3] ^= u[3];
        s[0] ^= u[0]; s[10] ^= u[0]; s[20] ^= u[0]; s[30] ^= u[0]; s[40] ^= u[0];
        s[1] ^= u[1]; s[11] ^= u[1]; s[21] ^= u[1]; s[31] ^= u[1]; s[41] ^= u[1];
        s[12] ^= u[2]; s[22] ^= u[2]; s[32] ^= u[2]; s[42] ^= u[2];
        s[13] ^= u[3]; s[23] ^= u[3]; s[33] ^= u[3]; s[43] ^= u[3];
        s[4] ^= u[4]; s[14] ^= u[4]; s[24] ^= u[4]; s[34] ^= u[4]; s[44] ^= u[4];
        s[5] ^= u[5]; s[15] ^= u[5]; s[25] ^= u[5]; s[35] ^= u[5]; s[45] ^= u[5];
        s[6] ^= u[6]; s[16] ^= u[6]; s[26] ^= u[6]; s[36] ^= u[6]; s[46] ^= u[6];
        s[7] ^= u[7]; s[17] ^= u[7]; s[27] ^= u[7]; s[37] ^= u[7]; s[47] ^= u[7];
        s[8] ^= u[8]; s[18] ^= u[8]; s[28] ^= u[8]; s[38] ^= u[8]; s[48] ^= u[8];
        s[9] ^= u[9]; s[19] ^= u[9]; s[29] ^= u[9]; s[39] ^= u[9]; s[49] ^= u[9];

        v[0] = s[2]; v[1] = s[3];
        ROTIs( 2, 12, 44);
        ROTIs(12, 18, 20);
        ROTIs(18, 44, 61);
        ROTIs(44, 28, 39);
        ROTIs(28, 40, 18);
        ROTIs(40,  4, 62);
        ROTIs( 4, 24, 43);
        ROTIs(24, 26, 25);
        ROTIs(26, 38,  8);
        ROTIs(38, 46, 56);
        ROTIs(46, 30, 41);
        ROTIs(30,  8, 27);
        ROTIs( 8, 48, 14);
        ROTIs(48, 42,  2);
        ROTIs(42, 16, 55);
        ROTIs(16, 32, 45);
        ROTIs(32, 10, 36);
        ROTIs(10,  6, 28);
        ROTIs( 6, 36, 21);
        ROTIs(36, 34, 15);
        ROTIs(34, 22, 10);
        ROTIs(22, 14,  6);
        ROTIs(14, 20,  3);
        ROTLI_1(s[20], s[21], v[0], v[1]);

        v[0] = s[ 0]; v[1] = s[ 2]; s[ 0] ^= (~s[ 2]) & s[ 4]; s[ 2] ^= (~s[ 4]) & s[ 6]; s[ 4] ^= (~s[ 6]) & s[ 8]; s[ 6] ^= (~s[ 8]) & v[0]; s[ 8] ^= (~v[0]) & v[1];
        v[0] = s[ 1]; v[1] = s[ 3]; s[ 1] ^= (~s[ 3]) & s[ 5]; s[ 3] ^= (~s[ 5]) & s[ 7]; s[ 5] ^= (~s[ 7]) & s[ 9]; s[ 7] ^= (~s[ 9]) & v[0]; s[ 9] ^= (~v[0]) & v[1];
        v[0] = s[10]; v[1] = s[12]; s[10] ^= (~s[12]) & s[14]; s[12] ^= (~s[14]) & s[16]; s[14] ^= (~s[16]) & s[18]; s[16] ^= (~s[18]) & v[0]; s[18] ^= (~v[0]) & v[1];
        v[0] = s[11]; v[1] = s[13]; s[11] ^= (~s[13]) & s[15]; s[13] ^= (~s[15]) & s[17]; s[15] ^= (~s[17]) & s[19]; s[17] ^= (~s[19]) & v[0]; s[19] ^= (~v[0]) & v[1];
        v[0] = s[20]; v[1] = s[22]; s[20] ^= (~s[22]) & s[24]; s[22] ^= (~s[24]) & s[26]; s[24] ^= (~s[26]) & s[28]; s[26] ^= (~s[28]) & v[0]; s[28] ^= (~v[0]) & v[1];
        v[0] = s[21]; v[1] = s[23]; s[21] ^= (~s[23]) & s[25]; s[23] ^= (~s[25]) & s[27]; s[25] ^= (~s[27]) & s[29]; s[27] ^= (~s[29]) & v[0]; s[29] ^= (~v[0]) & v[1];
        v[0] = s[30]; v[1] = s[32]; s[30] ^= (~s[32]) & s[34]; s[32] ^= (~s[34]) & s[36]; s[34] ^= (~s[36]) & s[38]; s[36] ^= (~s[38]) & v[0]; s[38] ^= (~v[0]) & v[1];
        v[0] = s[31]; v[1] = s[33]; s[31] ^= (~s[33]) & s[35]; s[33] ^= (~s[35]) & s[37]; s[35] ^= (~s[37]) & s[39]; s[37] ^= (~s[39]) & v[0]; s[39] ^= (~v[0]) & v[1];
        v[0] = s[40]; v[1] = s[42]; s[40] ^= (~s[42]) & s[44]; s[42] ^= (~s[44]) & s[46]; s[44] ^= (~s[46]) & s[48]; s[46] ^= (~s[48]) & v[0]; s[48] ^= (~v[0]) & v[1];
        v[0] = s[41]; v[1] = s[43]; s[41] ^= (~s[43]) & s[45]; s[43] ^= (~s[45]) & s[47]; s[45] ^= (~s[47]) & s[49]; s[47] ^= (~s[49]) & v[0]; s[49] ^= (~v[0]) & v[1];

        s[0] ^= d_RC[i];
        s[1] ^= d_RC[i+1];
    }
}

__global__ void quark_keccak512_interleaved_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *pHash = (uint32_t*)&g_hash[8 * hashPosition];
        uint32_t state[50];

        for (int i=0; i < 16; i += 2) {

            state[i] = pHash[i];
            state[i+1] = pHash[i+1];
            interleave(state[i], state[i+1]);
        }
        for (int i=18; i < 50; i++) {

            state[i] = 0;
        }

        state[16] = 1;
        state[17] = 0x80000000;

        keccak_interleaved_block(state);

        for( int i = 0; i < 16; i += 2 ) {

            deinterleave(state[i], state[i+1]);
            pHash[i] = state[i];
            pHash[i+1] = state[i+1];
        }
    }
}

__global__ void quark_keccak512_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *pHash = (uint32_t*)&g_hash[8 * hashPosition];
        uint32_t state[50];

#pragma unroll 8
        for (int i=0; i < 16; i += 2) {

            state[i] = pHash[i];
            state[i+1] = pHash[i+1];
        }
#pragma unroll 32
        for (int i=18; i < 50; i++) {

            state[i] = 0;
        }

        state[16] = 1;
        state[17] = 0x80000000;

        keccak_block(state);

#pragma unroll 8
        for( int i = 0; i < 16; i += 2 ) {

            pHash[i] = state[i];
            pHash[i+1] = state[i+1];
        }
    }
}

__global__ void quark_keccak512c_gpu_hash_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t state[50];

#pragma unroll 16
        for (int i=0; i < 16; i++) {

            state[i] = ((uint32_t *)g_hash)[i*threads+thread];
        }
#pragma unroll 32
        for (int i=18; i < 50; i++) {

            state[i] = 0;
        }

        state[16] = 1;
        state[17] = 0x80000000;

        keccak_block(state);

#pragma unroll 16
        for( int i = 0; i < 16; i++ ) {

            ((uint32_t *)g_hash)[i*threads+thread] = state[i];
        }
    }
}

__global__ void quark_keccak512_gpu_hash_final_64(int threads, uint32_t startNounce, uint64_t *g_hash, uint32_t *g_nonceVector, uint32_t *d_nonce)
{
    int thread = (blockDim.x * blockIdx.x + threadIdx.x);
    if (thread < threads)
    {
        uint32_t nounce = (g_nonceVector != NULL) ? g_nonceVector[thread] : (startNounce + thread);

        int hashPosition = nounce - startNounce;
        uint32_t *pHash = (uint32_t *)&g_hash[8 * hashPosition];
        uint32_t state[50];

#pragma unroll 8
        for (int i=0; i < 16; i += 2) {

            state[i] = pHash[i];
            state[i+1] = pHash[i+1];
        }
#pragma unroll 32
        for (int i=18; i < 50; i++) {

            state[i] = 0;
        }

        state[16] = 1;
        state[17] = 0x80000000;

        keccak_block(state);

		int position = -1;
		bool rc = true;

#pragma unroll 8
		for (int i = 7; i >= 0; i--) {
			if (state[i] > pTarget[i]) {
				if(position < i) {
					position = i;
					rc = false;
				}
	 		}
	 		if (state[i] < pTarget[i]) {
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
__host__ void quark_keccak512_cpu_init(int thr_id, int threads)
{
	cudaMalloc(&d_nonce[thr_id], sizeof(uint32_t));
    // Kopiere die Hash-Tabellen in den GPU-Speicher
    cudaMemcpyToSymbol(d_RC, h_RC, sizeof(h_RC), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_keccak512_interleaved_cpu_init(int thr_id, int threads)
{
    // Kopiere die Hash-Tabellen in den GPU-Speicher
    cudaMemcpyToSymbol(d_RC, h_interleaved_RC, sizeof(h_interleaved_RC), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_keccak512_cpu_setTarget(const void *ptarget)
{
	// die Message zur Berechnung auf der GPU
	cudaMemcpyToSymbol( pTarget, ptarget, 8*sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
}

__host__ void quark_keccak512_interleaved_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_keccak512_interleaved_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);
}

__host__ void quark_keccak512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_keccak512_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	cudaDeviceSynchronize();
}

__host__ void quark_keccak512c_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_keccak512c_gpu_hash_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t*)d_hash, d_nonceVector);

	cudaDeviceSynchronize();
}

__host__ uint32_t quark_keccak512_cpu_hash_final_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order)
{
    const int threadsperblock = 128;

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + threadsperblock-1)/threadsperblock);
    dim3 block(threadsperblock);

    cudaMemset(d_nonce[thr_id], 0xffffffff, sizeof(uint32_t));

    // Größe des dynamischen Shared Memory Bereichs
    size_t shared_size = 0;

    quark_keccak512_gpu_hash_final_64<<<grid, block, shared_size>>>(threads, startNounce, (uint64_t *)d_hash, d_nonceVector, d_nonce[thr_id]);

	cudaDeviceSynchronize();

    uint32_t res;
    cudaMemcpy(&res, d_nonce[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost);
    return res;
}
