spreadminer
===========

SpreadX11 miner based on Christian Buchner's &amp; Christian H.'s project ccminer.

Will code GPU miners for dust aka Donation addresses
====================================================
 * BTC: 1QD25HSCF8EAxUTYj2XsXZNGBi7RvQ21p8 [Yay, I'm BTC rich](https://blockchain.info/address/1QD25HSCF8EAxUTYj2XsXZNGBi7RvQ21p8)
 * SPR: SfSEcVQGhbXvPQ2hkTj3vxSd9PEZA12efa [Yay, I'm SPR rich](http://spreadcoin.net/explorer/index.php?q=SfSEcVQGhbXvPQ2hkTj3vxSd9PEZA12efa)

Requirements
============
 * GPU with compute capability 3.0+

Dependencies
============
 * libcurl: http://curl.haxx.se/libcurl/
 * jansson: http://www.digip.org/jansson/ (jansson is included in-tree)
 * openssl: https://www.openssl.org/

Download
========
 * Binary releases: https://github.com/tsiv/spreadminer/releases

Build
=====

#### Linux
 * edit the NVCC_GENCODE line in Makefile.am to target your preferred compute capability
 * ./autogen.sh
 * ./configure "CFLAGS=-O3" "CXXFLAGS=-O3"
 * make

#### Windows
 * Go to http://cudamining.co.uk/url/tutorials/id/3 and follow the instructions
 * Pray that it works and that you don't go crazy trying to scrape together the dependencies

Usage
=====
 * Quick start: spreadminer -o http://127.0.0.1:41677 -u rpcuser -p rpcpass (obviously replace 127.0.0.1 if your wallet is running on another machine)
 * spreadminer -h for additional options (some of them might be leftovers from ccminer and not work)
 * experiment with the -x parameter to find a value that gives you best performance and doesn't lag out your desktop too much