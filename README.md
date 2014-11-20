spreadminer
===========

SpreadX11 miner based on Christian Buchner's &amp; Christian H.'s project ccminer.

Requirements
============
 * GPU with compute capability 3.2+

Dependencies
============
 * libcurl: http://curl.haxx.se/libcurl/
 * jansson: http://www.digip.org/jansson/ (jansson is included in-tree)
 * openssl: https://www.openssl.org/

Download
========
 * Binary releases: https://github.com/LucasJones/cpuminer-multi/releases
 * Git tree:   https://github.com/LucasJones/cpuminer-multi
 * Clone with `git clone https://github.com/LucasJones/cpuminer-multi`

Build
=====

#### Linux
 * edit the NVCC_GENCODE line in Makefile.am to target your preferred compute capability
 * ./autogen.sh
 * ./configure CFLAGS="-O3"
 * make

#### Windows
 * Go to http://cudamining.co.uk/url/tutorials/id/3 and follow the instructions
 * Pray that it works

Usage
=====
 * Quick start: spreadminer -o http://127.0.0.1:41677 -u rpcuser -p rpcpass (obviously replace 127.0.0.1 if your wallet is running on another machine)
 * spreadminer -h for additional options (some of them might be leftovers from ccminer and not work)
 * experiment with the -x parameter to find a value that gives you best performance and doesn't lag out your desktop too much