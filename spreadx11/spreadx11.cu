#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "miner.h"

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
    printf("%s: %.2f sec, %.2f MH/s\n", s, duration, (double)throughput / 1000000.0 / duration); \
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

            applog(LOG_DEBUG, 
                "Thread %d found a solution\n"
                "First nonce : %08x\n"
                "Found nonce : %08x\n"
                "Threadidx   : %d\n"
                "Threadidx64 : %d\n"
                "VTX         : %s\n",
                thr_id, first_nonce, foundNonce, winnerthread, idx64, hexbuffer);

            *resnonce = foundNonce;
            cudaMemcpy(reshashwholeblock, d_hashwholeblock[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
            cudaMemcpy(ressignature, d_signature[thr_id] + idx64 * 8, 32, cudaMemcpyDeviceToHost);
            cudaMemcpy(hash, d_hash[thr_id] + winnerthread * 16, 32, cudaMemcpyDeviceToHost);

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

            // FIXME: should probably implement a CPU version to check the hash before submitting
            if( fulltest(hash, ptarget) ) {
                
                //*hashes_done = foundNonce - first_nonce + 1;
	            *hashes_done = nonce + throughput - first_nonce + 1;
                return 1;
            }
		}

		nonce += throughput;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);

	*hashes_done = nonce - first_nonce + 1;
	return 0;
}
