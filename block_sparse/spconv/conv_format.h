#pragma once


enum LayoutFilter {
    filter_channel_first,  // F-R-S-C
    filter_planar_first    // F-C-R-S
};

// enum LayoutFmap {
//     fmap_batch_first,    // C-H-W-B
//     fmap_planar_first,   // B-C-H-W
// };

template<LayoutFilter layout_filter>
__device__ __forceinline__ int col2im(
    int B, int H, int W, int C, int R, int S, int id_row, int id_col) 
{
    int cc, rr, ss;
    if constexpr (layout_filter == filter_channel_first) {
        cc = id_row % C;
        ss = (id_row / C) % S;
        rr = (id_row / C) / S;
    }
    else {
        ss = (id_row % S);
        rr = (id_row / S) % R;
        cc = (id_row / S) / R;
    }
    int nn, wo, ho;
    // if constexpr (layout_fmap == fmap_batch_first) {
    if (true) { // always use batch-innermost
        nn = id_col % B;
        wo = (id_col / B) % W;
        ho = (id_col / B) / W;
    }
    else {
        wo = (id_col % W);
        ho = (id_col / W) % H;
        nn = (id_col / W) / H;
    }
    
    int wi = wo - (S/2) + ss;
    int hi = ho - (R/2) + rr;
    if ((wi >= 0 && wi < W) && (hi >= 0 && hi < H))
        return nn + (wi + (hi + cc * H) * W) * B;
    return -1;
}

template<LayoutFilter layout_filter>
__device__ __forceinline__ int col2im(
    int B, int H, int W, int C, int R, int S, int id_row, int id_col, int stride) 
{
    int cc, rr, ss;
    if constexpr (layout_filter == filter_channel_first) {
        cc = id_row % C;
        ss = (id_row / C) % S;
        rr = (id_row / C) / S;
    }
    else {
        ss = (id_row % S);
        rr = (id_row / S) % R;
        cc = (id_row / S) / R;
    }
    int nn, wo, ho;
    // if constexpr (layout_fmap == fmap_batch_first) {
    if (true) { // always use batch-innermost
        nn = id_col % B;
        wo = (id_col / B) % W;
        ho = (id_col / B) / W;
    }
    else {
        wo = (id_col % W);
        ho = (id_col / W) % H;
        nn = (id_col / W) / H;
    }
    
    int wi = wo*stride - (S/2) + ss;
    int hi = ho*stride - (R/2) + rr;
    if ((wi >= 0 && wi < W) && (hi >= 0 && hi < H))
        return nn + (wi + (hi + cc * H) * W) * B;
    return -1;
}
