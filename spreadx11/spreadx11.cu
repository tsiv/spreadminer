#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

extern "C"
{
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_groestl.h"
#include "sph/sph_skein.h"
#include "sph/sph_jh.h"
#include "sph/sph_keccak.h"

#include "sph/sph_luffa.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_shavite.h"
#include "sph/sph_simd.h"
#include "sph/sph_echo.h"

#include <openssl/bn.h>
#include <openssl/sha.h>

#include "miner.h"
}

extern "C" int device_map[8];
extern "C" bool opt_benchmark;
extern "C" int opt_throughput;

static uint32_t *d_sha256hash[8];
static uint32_t *d_signature[8];
static uint32_t *d_hashwholeblock[8];
static uint32_t *d_hash[8];
static uint32_t *d_wholeblockdata[8];

extern void spreadx11_sha256double_cpu_hash_88(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash);
extern void spreadx11_sha256double_setBlock_88(void *data);
extern void spreadx11_sha256_cpu_init( int thr_id, int throughput );

extern void spreadx11_sha256_cpu_hash_wholeblock(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_wholeblock);
extern void spreadx11_sha256_setBlock_wholeblock( struct work *work, uint32_t *d_wholeblock );

extern void spreadx11_sign_cpu_init( int thr_id, int throughput );
extern void spreadx11_sign_cpu_setInput( struct work *work );
extern void spreadx11_sign_cpu(int thr_id, int threads, uint32_t startNonce, uint32_t *d_hash, uint32_t *d_signature);

extern void blake_cpu_init(int thr_id, int threads);
extern void blake_cpu_setBlock_185(void *pdata);
extern void blake_cpu_hash_185(int thr_id, int threads, uint32_t startNounce, uint32_t *d_hash, uint32_t *d_signature, uint32_t *d_hashwholeblock);

extern void quark_bmw512_cpu_init(int thr_id, int threads);
extern void quark_bmw512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_groestl512_cpu_init(int thr_id, int threads);
extern void quark_groestl512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_skein512_cpu_init(int thr_id, int threads);
extern void quark_skein512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_keccak512_cpu_init(int thr_id, int threads);
extern void quark_keccak512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void quark_jh512_cpu_init(int thr_id, int threads);
extern void quark_jh512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_luffa512_cpu_init(int thr_id, int threads);
extern void x11_luffa512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_cubehash512_cpu_init(int thr_id, int threads);
extern void x11_cubehash512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_shavite512_cpu_init(int thr_id, int threads);
extern void x11_shavite512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_simd512_cpu_init(int thr_id, int threads);
extern void x11_simd512_cpu_hash_64(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int order);

extern void x11_echo512_cpu_init(int thr_id, int threads);
extern uint32_t x11_echo512_cpu_hash_64_final(int thr_id, int threads, uint32_t startNounce, uint32_t *d_nonceVector, uint32_t *d_hash, int *hashidx);
extern void x11_echo512_cpu_setTarget(const void *ptarget);

#define PROFILE 0
#if PROFILE == 1
#define PRINTTIME(s) do { \
    double duration; \
    gettimeofday(&tv_end, NULL); \
    duration = 1e-6 * (tv_end.tv_usec-tv_start.tv_usec) + (tv_end.tv_sec-tv_start.tv_sec); \
    printf("%s: %.2f ms, %.2f MH/s\n", s, duration*1000.0, (double)throughput / 1000000.0 / duration); \
    } while(0)
#else
#define PRINTTIME(s)
#endif

void hextobin(unsigned char *p, const char *hexstr, size_t len)
{
	char hex_byte[3];
	char *ep;

	hex_byte[2] = '\0';

	while (*hexstr && len) {
		if (!hexstr[1]) {
			applog(LOG_ERR, "hex2bin str truncated");
			return;
		}
		hex_byte[0] = hexstr[0];
		hex_byte[1] = hexstr[1];
		*p = (unsigned char) strtol(hex_byte, &ep, 16);
		if (*ep) {
			applog(LOG_ERR, "hex2bin failed on '%s'", hex_byte);
			return;
		}
		p++;
		hexstr += 2;
		len--;
	}
}

extern "C" void spreadx11_hash( void *output, struct work *work, uint32_t nonce )
{
    SHA256_CTX ctx_sha;
    sph_blake512_context ctx_blake;
    sph_bmw512_context ctx_bmw;
    sph_groestl512_context ctx_groestl;
    sph_jh512_context ctx_jh;
    sph_keccak512_context ctx_keccak;
    sph_skein512_context ctx_skein;
    sph_luffa512_context ctx_luffa;
    sph_cubehash512_context ctx_cubehash;
    sph_shavite512_context ctx_shavite;
    sph_simd512_context ctx_simd;
    sph_echo512_context ctx_echo;

    unsigned char mod[32] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe, 
        0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b, 0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36, 0x41, 0x41
    };
    unsigned char tmp[185];
    unsigned char hashforsig[32];
    unsigned char wholeblock[MAX_BLOCK_SIZE];
    unsigned char finalhash[64];
    BIGNUM *bn_hash, *bn_privkey, *bn_kinv, *bn_mod, *bn_res;
    BN_CTX *bn_ctx;

    uint32_t *nonceptr = (uint32_t *)&tmp[84];
    unsigned char *hashwholeblockptr = &tmp[88];
    unsigned char *signatureptr = &tmp[153];

    memcpy(tmp, work->data, 185);
    *nonceptr = nonce & 0xffffffc0;
    
    SHA256_Init(&ctx_sha);
    SHA256_Update(&ctx_sha, tmp, 88);
    SHA256_Final(hashforsig, &ctx_sha);
    SHA256_Init(&ctx_sha);
    SHA256_Update(&ctx_sha, hashforsig, 32);
    SHA256_Final(hashforsig, &ctx_sha);

    bn_ctx = BN_CTX_new();
    bn_hash = BN_new();
    bn_privkey = BN_new();
    bn_kinv = BN_new();
    bn_mod = BN_new();
    bn_res = BN_new();

    BN_bin2bn(hashforsig, 32, bn_hash);
    BN_bin2bn(work->privkey, 32, bn_privkey);
    BN_bin2bn(work->kinv, 32, bn_kinv);
    BN_bin2bn(mod, 32, bn_mod);

    BN_mod_add_quick(bn_privkey, bn_privkey, bn_hash, bn_mod);
    BN_mod_mul(bn_res, bn_privkey, bn_kinv, bn_mod, bn_ctx);
    int nBitsS = BN_num_bits(bn_res);
    memset(signatureptr, 0, 32);
    BN_bn2bin(bn_res, &signatureptr[32-(nBitsS+7)/8]);

    BN_CTX_free(bn_ctx);
    BN_clear_free(bn_hash);
    BN_clear_free(bn_privkey);
    BN_clear_free(bn_kinv);
    BN_clear_free(bn_mod);
    BN_clear_free(bn_res);

    memcpy(wholeblock+0, tmp+84, 4); // nNonce
    memcpy(wholeblock+4, tmp+68, 8); // nTime
    memcpy(wholeblock+12, tmp+120, 65); // MinerSignature
    memcpy(wholeblock+77, tmp+0, 4); // nVersion
    memcpy(wholeblock+81, tmp+4, 32); // hashPrevBlock
    memcpy(wholeblock+113, tmp+36, 32); // HashMerkleRoot
    memcpy(wholeblock+145, tmp+76, 4); // nBits
    memcpy(wholeblock+149, tmp+80, 4); // nHeight
    memcpy(wholeblock+153, work->tx, work->txsize); // tx

    // the total amount of bytes in our data
    int blocksize = 153 + work->txsize;

    // pad the block with 0x07 bytes to make it a multiple of uint32_t
    while( blocksize & 3 ) wholeblock[blocksize++] = 0x07;

    // figure out the offsets for the padding
    uint32_t *pFillBegin = (uint32_t*)&wholeblock[blocksize];
    uint32_t *pFillEnd = (uint32_t*)&wholeblock[MAX_BLOCK_SIZE]; // FIXME: isn't this out of bounds by one... but it seems to work out...
    uint32_t *pFillFooter = pFillBegin > pFillEnd - 8 ? pFillBegin : pFillEnd - 8;

    memcpy(pFillFooter, tmp+4, (pFillEnd - pFillFooter)*4);
    for (uint32_t *pI = pFillFooter; pI < pFillEnd; pI++)
        *pI |= 1;

    for (uint32_t *pI = pFillFooter - 1; pI >= pFillBegin; pI--)
        pI[0] = pI[3]*pI[7];

    SHA256_Init(&ctx_sha);
    SHA256_Update(&ctx_sha, wholeblock, MAX_BLOCK_SIZE);
    SHA256_Update(&ctx_sha, wholeblock, MAX_BLOCK_SIZE);
    SHA256_Final(hashwholeblockptr, &ctx_sha);

    *nonceptr = nonce;

    sph_blake512_init(&ctx_blake);
    sph_blake512 (&ctx_blake, tmp, 185);
    sph_blake512_close(&ctx_blake, (void*) finalhash);
    
    sph_bmw512_init(&ctx_bmw);
    sph_bmw512 (&ctx_bmw, (const void*) finalhash, 64);
    sph_bmw512_close(&ctx_bmw, (void*) finalhash);

    sph_groestl512_init(&ctx_groestl);
    sph_groestl512 (&ctx_groestl, (const void*) finalhash, 64);
    sph_groestl512_close(&ctx_groestl, (void*) finalhash);

    sph_skein512_init(&ctx_skein);
    sph_skein512 (&ctx_skein, (const void*) finalhash, 64);
    sph_skein512_close(&ctx_skein, (void*) finalhash);

    sph_jh512_init(&ctx_jh);
    sph_jh512 (&ctx_jh, (const void*) finalhash, 64);
    sph_jh512_close(&ctx_jh, (void*) finalhash);

    sph_keccak512_init(&ctx_keccak);
    sph_keccak512 (&ctx_keccak, (const void*) finalhash, 64);
    sph_keccak512_close(&ctx_keccak, (void*) finalhash);

    sph_luffa512_init(&ctx_luffa);
    sph_luffa512 (&ctx_luffa, (const void*) finalhash, 64);
    sph_luffa512_close (&ctx_luffa, (void*) finalhash);

    sph_cubehash512_init(&ctx_cubehash);
    sph_cubehash512 (&ctx_cubehash, (const void*) finalhash, 64);
    sph_cubehash512_close(&ctx_cubehash, (void*) finalhash);

    sph_shavite512_init(&ctx_shavite);
    sph_shavite512 (&ctx_shavite, (const void*) finalhash, 64);
    sph_shavite512_close(&ctx_shavite, (void*) finalhash);

    sph_simd512_init(&ctx_simd);
    sph_simd512 (&ctx_simd, (const void*) finalhash, 64);
    sph_simd512_close(&ctx_simd, (void*) finalhash);

    sph_echo512_init(&ctx_echo);
    sph_echo512 (&ctx_echo, (const void*) finalhash, 64);
    sph_echo512_close(&ctx_echo, (void*) finalhash); 

    memcpy(output, finalhash, 32);
}

extern "C" int scanhash_spreadx11( int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done )
{
    // multiple of 64 to keep things simple with signatures
    const int throughput = opt_throughput * 1024 * 64;
    unsigned char *blocktemplate = work->data;
    uint32_t *ptarget = work->target;
    uint32_t *pnonce = (uint32_t *)&blocktemplate[84];
    uint32_t nonce = *pnonce;
    uint32_t first_nonce = nonce;
    
    if (opt_benchmark)
        ((uint32_t*)ptarget)[7] = 0x000000ff;

    static bool init[8] = {0,0,0,0,0,0,0,0};
    if (!init[thr_id])
    {
        cudaSetDevice(device_map[thr_id]);
        cudaDeviceReset();
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		
		// sha256 hashes used for signing, 32 bytes for every 64 nonces
        cudaMalloc(&d_sha256hash[thr_id], 32*(throughput>>6));
        // changing part of MinerSignature, 32 bytes for every 64 nonces
        cudaMalloc(&d_signature[thr_id], 32*(throughput>>6));
        // sha256 hashes for the whole block, 32 bytes for every 64 nonces
		cudaMalloc(&d_hashwholeblock[thr_id], 32*(throughput>>6));
        // single buffer to hold the padded whole block data
        cudaMalloc(&d_wholeblockdata[thr_id], 200000);
		// a 512-bit buffer for every nonce to hold the x11 intermediate hashes
        cudaMalloc(&d_hash[thr_id], 16 * sizeof(uint32_t) * throughput);

        spreadx11_sha256_cpu_init(thr_id, throughput);
        spreadx11_sign_cpu_init(thr_id, throughput);
        blake_cpu_init(thr_id, throughput);
		quark_groestl512_cpu_init(thr_id, throughput);
		quark_skein512_cpu_init(thr_id, throughput);
		quark_bmw512_cpu_init(thr_id, throughput);
		quark_keccak512_cpu_init(thr_id, throughput);
		quark_jh512_cpu_init(thr_id, throughput);
		x11_luffa512_cpu_init(thr_id, throughput);
		x11_cubehash512_cpu_init(thr_id, throughput);
		x11_shavite512_cpu_init(thr_id, throughput);
		x11_simd512_cpu_init(thr_id, throughput);
		x11_echo512_cpu_init(thr_id, throughput);

        init[thr_id] = true;
    }

    struct timeval tv_start, tv_end;

    spreadx11_sign_cpu_setInput(work);
    spreadx11_sha256_setBlock_wholeblock(work, d_wholeblockdata[thr_id]);
    spreadx11_sha256double_setBlock_88((void *)blocktemplate);
    blake_cpu_setBlock_185((void *)blocktemplate);
    x11_echo512_cpu_setTarget(ptarget);

	do {
		int order = 0;

        gettimeofday(&tv_start, NULL);
        spreadx11_sha256double_cpu_hash_88(thr_id, throughput>>6, nonce, d_sha256hash[thr_id]);
        PRINTTIME("sha256 for signature");

		gettimeofday(&tv_start, NULL);
		spreadx11_sign_cpu(thr_id, throughput>>6, nonce, d_sha256hash[thr_id], d_signature[thr_id]);
        PRINTTIME("signing");

		gettimeofday(&tv_start, NULL);
		spreadx11_sha256_cpu_hash_wholeblock(thr_id, throughput>>6, nonce, d_hashwholeblock[thr_id], d_signature[thr_id], d_wholeblockdata[thr_id]);
        PRINTTIME("hashwholeblock");

		gettimeofday(&tv_start, NULL);
		blake_cpu_hash_185(thr_id, throughput, nonce, d_hash[thr_id], d_signature[thr_id], d_hashwholeblock[thr_id]);
        PRINTTIME("blake");

		gettimeofday(&tv_start, NULL);
		quark_bmw512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("bmw");

		gettimeofday(&tv_start, NULL);
		quark_groestl512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("groestl");

		gettimeofday(&tv_start, NULL);
		quark_skein512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("skein");

		gettimeofday(&tv_start, NULL);
		quark_jh512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("jh");

		gettimeofday(&tv_start, NULL);
		quark_keccak512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("keccak");

		gettimeofday(&tv_start, NULL);
		x11_luffa512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("luffa");

		gettimeofday(&tv_start, NULL);
		x11_cubehash512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("cubehash");

		gettimeofday(&tv_start, NULL);
		x11_shavite512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("shavite");

		gettimeofday(&tv_start, NULL);
		x11_simd512_cpu_hash_64(thr_id, throughput, nonce, NULL, d_hash[thr_id], order++);
        PRINTTIME("simd");

		
        int winnerthread;
        uint32_t foundNonce;

        gettimeofday(&tv_start, NULL);
		foundNonce = x11_echo512_cpu_hash_64_final(thr_id, throughput, nonce, NULL, d_hash[thr_id], &winnerthread);
        PRINTTIME("echo");

		if  (foundNonce != 0xffffffff)
		{
            uint32_t hash[8];
            char hexbuffer[MAX_BLOCK_SIZE*2];
            
            memset(hexbuffer, 0, sizeof(hexbuffer));
            
            for( int i = 0; i < work->txsize && i < MAX_BLOCK_SIZE; i++ ) sprintf(&hexbuffer[i*2], "%02x", work->tx[i]);

            uint32_t *resnonce = (uint32_t *)&work->data[84];
            uint32_t *reshashwholeblock = (uint32_t *)&work->data[88];
            uint32_t *ressignature = (uint32_t *)&work->data[153];
            uint32_t idx64 = winnerthread >> 6;

            if( opt_debug ) 
                applog(LOG_DEBUG, 
                    "Thread %d found a solution\n"
                    "First nonce : %08x\n"
                    "Found nonce : %08x\n"
                    "Threadidx   : %d\n"
                    "Threadidx64 : %d\n"
                    "VTX         : %s\n",
                    thr_id, first_nonce, foundNonce, winnerthread, idx64, hexbuffer);
            else applog(LOG_INFO, "GPU #%d: found a solution, nonce $%08X", thr_id, foundNonce);


            *resnonce = foundNonce;
            cudaMemcpy(reshashwholeblock, d_hashwholeblock[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
            cudaMemcpy(ressignature, d_signature[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
            cudaMemcpy(hash, d_hash[thr_id] + winnerthread * 16, 32, cudaMemcpyDeviceToHost);

            if( opt_debug ) {

                memset(hexbuffer, 0, sizeof(hexbuffer));
                for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)hash)[i]);
                applog(LOG_DEBUG, "Final hash 256 : %s", hexbuffer);

                memset(hexbuffer, 0, sizeof(hexbuffer));
                for( int i = 0; i < 185; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)work->data)[i]);
                applog(LOG_DEBUG, "Submit data    : %s", hexbuffer);
                
                memset(hexbuffer, 0, sizeof(hexbuffer));
                for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)reshashwholeblock)[i]);
                applog(LOG_DEBUG, "HashWholeBlock : %s", hexbuffer);

                memset(hexbuffer, 0, sizeof(hexbuffer));
                for( int i = 0; i < 32; i++ ) sprintf(&hexbuffer[i*2], "%02x", ((uint8_t *)ressignature)[i]);
                applog(LOG_DEBUG, "MinerSignature : %s", hexbuffer);
            }

            uint32_t cpuhash[8];
            spreadx11_hash((void *)cpuhash, work, foundNonce);

            if( cpuhash[7] == hash[7] && fulltest(hash, ptarget) ) {
                
                //*hashes_done = foundNonce - first_nonce + 1;
	            *hashes_done = nonce + throughput - first_nonce + 1;
                return 1;
			} else {
				applog(LOG_INFO, "GPU #%d: result for nonce $%08X does not validate on CPU!", thr_id, foundNonce);
			}
		}

		nonce += throughput;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = nonce - first_nonce + 1;
	return 0;
}
