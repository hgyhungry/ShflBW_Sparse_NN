#pragma once

#include <string>
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <random>       
#include <chrono>       // timeseed


void parseSpmmArgs(const int argc, const char** argv, 
    int &m, int &n, int &k, float &density, unsigned &seed, 
    int &pattern_code, int &block_sz, bool &row_permute,
    bool &store_pattern, std::string &store_path, 
    bool &load_pattern,  std::string &load_path, bool verbose=false)
{
    std::string Usage =     
        "\tRequired cmdline args:\n\
        --m [M]\n\
        --n [N]\n\
        --k [K]\n\
        --d [density between 0~1]\n\
        --sparsity-type [block/block-2in4] \n\
        --block-size [16/32/64/128] \n\
        Optional cmdline args: \n\
        --row-permute : run spmm with random row-permutation \n\
        --random: set a random seed. by default 2021 for every run.\n\
        --store-pattern [file]: store the random sparse pattern generated,\n\
        --load-pattern  [file]: load the random sparse pattern from file\n\
    \n";
    // default
    m = 0; n = 0; k = 0; density = 0.f; seed = 2021;
    block_sz = 0; pattern_code = -1;
    row_permute = false;
    store_pattern = false; load_pattern = false;
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--m") && i!=argc-1) {
            m = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--n") && i!=argc-1) {
            n = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--k") && i!=argc-1) {
            k = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--d") && i!=argc-1) {
            density = atof(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--block-size") && i!=argc-1) {
            block_sz = atof(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--sparsity-type") && i!=argc-1) {
            if (!strcmp(argv[i+1], "block")) 
                pattern_code = 0;
            else if (!strcmp(argv[i+1], "block-2in4")) 
                pattern_code = 1;
        }
        else if (!strcmp(argv[i], "--row-permute") ) {
            row_permute = true;
        }
        else if (!strcmp(argv[i], "--random") ) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
        else if (!strcmp(argv[i], "--store-pattern") && i!=argc-1) {
            store_pattern = true;
            store_path = std::string(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--load-pattern") && i!=argc-1) {
            load_pattern = true;
            load_path = std::string(argv[i+1]);
        }
    }

    std::stringstream log;
    if (load_pattern) {
        // load_blocksparse_banner(load_path.c_str(), m, k, density, seed, 
            // pattern_code, block_sz, row_permute);      
        std::cerr << "load pattern not implemented.\n";
        exit(EXIT_FAILURE);
    }
    if (load_pattern) 
        log <<    "load path = " << load_path ;

    log     << "\narguments: m = " << m
            <<            "\nn = " << n
            <<            "\nk = " << k
            <<      "\ndensity = " << density 
            <<         "\nseed = " << seed 
            <<   "\nblock-size = " << block_sz << " x 1"
            << "\nsparsity-pattern = " << pattern_code
            <<  "\nrow-permute ? " << row_permute;

    if (store_pattern) 
        log <<   "\nstore path = " << store_path;
    log     <<   "\n" ;

    if (m == 0 || n==0 || k==0 || density==0.f ) {
        std::cerr << Usage;
        std::cerr << "Forget to set m,n,k or density? \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (!( (block_sz & (block_sz-1))==0 && block_sz >=16 )) {
        std::cerr << Usage;
        std::cerr << "Unsupported block size. Choose in 16/32/64/128. \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (pattern_code == -1) {
        std::cerr << Usage;
        std::cerr << "sparsity-pattern is not given or unsupported.\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    
    if (verbose) {
        std::cerr << log.str();
    }
}

void parseSpconvArgs(const int argc, const char** argv, 
    int &B, int &H, int &W, int &C, int &F, int &R, int &S, int &stride,
    float &density, unsigned &seed, int &pattern_code, int &block_sz, 
    bool &row_permute,
    bool &store_pattern, std::string &store_path, 
    bool &load_pattern,  std::string &load_path, bool verbose=false)
{
    std::string Usage =     
        "\tRequired cmdline args:\n\
        --B [B]\n\
        --H [H]\n\
        --W [W]\n\
        --C [C]\n\
        --F [F]\n\
        --R [R]\n\
        --S [S]\n\
        --stride [stride]\n\
        --d [density between 0~1]\n\
        --sparsity-type [block/block-2in4] \n\
        --block-size [16/32/64/128] \n\
        Optional cmdline args: \n\
        --row-permute : run spmm with random row-permutation \n\
        --random: set a random seed. by default 2021 for every run.\n\
        --store-pattern [file]: store the random sparse pattern generated,\n\
        --load-pattern  [file]: load the random sparse pattern from file\n\
    \n";
    // default
    B = 0; H = 0; W = 0; C = 0; F = 0; R = 0; S = 0; stride = 1;
    density = 0.f; seed = 2021;
    block_sz = 0; pattern_code = -1;
    row_permute = false;
    store_pattern = false; load_pattern = false;
    
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--B") && i!=argc-1) {
            B = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--H") && i!=argc-1) {
            H = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--W") && i!=argc-1) {
            W = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--C") && i!=argc-1) {
            C = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--F") && i!=argc-1) {
            F = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--R") && i!=argc-1) {
            R = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--S") && i!=argc-1) {
            S = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--stride") && i!=argc-1) {
            stride = atoi(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--d") && i!=argc-1) {
            density = atof(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--block-size") && i!=argc-1) {
            block_sz = atof(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--sparsity-type") && i!=argc-1) {
            if (!strcmp(argv[i+1], "block")) 
                pattern_code = 0;
            else if (!strcmp(argv[i+1], "block-2in4")) 
                pattern_code = 1;
        }
        else if (!strcmp(argv[i], "--row-permute") ) {
            row_permute = true;
        }
        else if (!strcmp(argv[i], "--random") ) {
            seed = std::chrono::system_clock::now().time_since_epoch().count();
        }
        else if (!strcmp(argv[i], "--store-pattern") && i!=argc-1) {
            store_pattern = true;
            store_path = std::string(argv[i+1]);
        }
        else if (!strcmp(argv[i], "--load-pattern") && i!=argc-1) {
            load_pattern = true;
            load_path = std::string(argv[i+1]);
        }
    }

    std::stringstream log;
    if (load_pattern) {
        // load_blocksparse_banner(load_path.c_str(), m, k, density, seed, 
            // pattern_code, block_sz, row_permute);      
        std::cerr << "load pattern not implemented.\n";
        exit(EXIT_FAILURE);
    }
    if (load_pattern) 
        log <<    "load path = " << load_path ;

    log     << "\narguments: B = " << B
            <<            "\nH = " << H
            <<            "\nW = " << W
            <<            "\nC = " << C
            <<            "\nF = " << F
            <<            "\nR = " << R
            <<            "\nS = " << S
            <<       "\nstride = " << stride
            <<      "\ndensity = " << density 
            <<         "\nseed = " << seed 
            <<   "\nblock-size = " << block_sz << " x 1"
            << "\nsparsity-pattern = " << pattern_code
            <<  "\nrow-permute ? " << row_permute;

    if (store_pattern) 
        log <<   "\nstore path = " << store_path;
    log     <<   "\n" ;

    if (B == 0 || H==0 || W==0 || C ==0 ||
        F == 0 || R==0 || S==0 ||density==0.f || stride<0) {
        std::cerr << Usage;
        std::cerr << "Forget to set m,n,k or density\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (stride<0) {
        std::cerr << Usage;
        std::cerr << "Negative stride is not allowed.\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (!( (block_sz & (block_sz-1))==0 && block_sz >=16 )) {
        std::cerr << Usage;
        std::cerr << "Unsupported block size. Choose in 16/32/64/128. \n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    if (pattern_code == -1) {
        std::cerr << Usage;
        std::cerr << "sparsity-pattern is not given or unsupported.\n";
        std::cerr << log.str();
        exit(EXIT_FAILURE);
    }
    
    if (verbose) {
        std::cerr << log.str();
    }
}