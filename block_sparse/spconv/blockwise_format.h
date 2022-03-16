#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <iterator>             // for generating row_indices
#include <algorithm>
#include <random>
#include <chrono>
#include "block_sparse/common/base.h"        // ROUND_UP, CEIL
#include "block_sparse/cuda_error.h"         // CUDA_CHECK
#include "block_sparse/util/random_mask.h"

#include "conv_format.h"                    // layout enums
#include "block_sparse/spmm/blockwise_format.h"

template<typename T>
struct BlockwiseSpFilter 
: public BlockwiseSpMatrix<T> {
    // shape and pattern config
    int F;  // number of filters
    int C;  // number of input channels
    int R;  // number of rows
    int S;  // number of columns

    void init_random(int F, int R, int S, int C, int brow, int bcol, 
    float expected_density, bool filter_permute, unsigned seed);
};

template<typename T>
void BlockwiseSpFilter<T>::init_random(int F_, int R_, int S_, int C_, 
int brow, int bcol, float expected_density, bool filter_permute, unsigned seed
) {
    this->F = F_;
    this->R = R_;
    this->S = S_;
    this->C = C_;

    BlockwiseSpMatrix<T>::init_random(F_, R_*S_*C_, brow, bcol, expected_density,
    filter_permute, seed);

    // generate a config string for logging
    std::stringstream s;
    s << F_ << " " << R_ << " " << S_ << " " << C_ << " "
      << brow << " " << bcol << " " << this->density << " " << seed << " ";
    if (this->row_permute) {
        s << "filter-permute ";
    }
    else {
        s << "non-filter-permute ";
    }
    this->config_str = s.str();
}

template<typename T>
void get_host_reference(BlockwiseSpFilter<T> &spfilter, const int B, const int H,
const int W, const int stride, const std::vector<T> &IFMap, 
std::vector<float> &OFMap_ref, LayoutFilter layout_filter) {
    int brow = spfilter.brow; 
    int bcol = spfilter.bcol;
    int F    = spfilter.F;
    int C    = spfilter.C;
    int R    = spfilter.R;
    int S    = spfilter.S;

    for (int id = 0; id < B*F*H*W; id++) 
        OFMap_ref[id] = 0.f;
    
    for (int i = 0; i < F/brow; i++) {
        int begin = spfilter.indptr[i];
        int end   = spfilter.indptr[i+1];
        auto itF  = spfilter.values.begin() + begin * brow * bcol;
        for (int p = begin; p < end; p++, itF += brow*bcol) {
            int j = spfilter.indices[p] * bcol;
            for (int kk = 0; kk < bcol; kk++) {
                int k = j+kk;
                int cc, ss, rr;
                if (layout_filter == filter_channel_first) {
                    cc = k % C;
                    ss = (k / C) % S;
                    rr = (k / C) / S;
                }
                else {
                    ss = k % S;
                    rr = (k / S) % R;
                    cc = (k / S) / R;
                }
                
                for (int x = 0; x < brow; x++) {
                    int ff = i*brow + x;
                    if (spfilter.row_permute) {
                        ff = spfilter.row_permute_ids[ff];
                    }

                    float w = static_cast<float>(*(itF + kk*brow + x));

                    for (int y = 0; y < B*H*W; y++) {

                        int nn, wo, ho;
                        // if (layout_fmap == fmap_batch_first) {
                        if (true) {
                            nn = y % B;
                            wo = (y / B) % W;
                            ho = (y / B) / W;
                        }
                        else {
                            wo = y % W;
                            ho = (y / W) % H;
                            nn = (y / W) / H;
                        }
                        
                        int hi = ho*stride - (R/2) + rr;
                        int wi = wo*stride - (S/2) + ss;
                        
                        float i;
                        if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                            int ifmap_idx;
                            // if (layout_fmap == fmap_batch_first)
                            if (true)
                                ifmap_idx = nn + (wi + (hi + cc*H)*W)*B;
                            else
                                ifmap_idx = wi + (hi + (nn + cc*B)*H)*W;
                            i = static_cast<float>(IFMap[ifmap_idx]);
                        }
                        else 
                            i = 0.f;
                        // if (layout_fmap == fmap_batch_first)
                        if (true) 
                            OFMap_ref[nn + (wo + (ho + ff*H)*W)*B] += (w*i);
                        else 
                            OFMap_ref[wo + (ho + (nn + ff*B)*H)*W] += (w*i);
    }}}}}
}
