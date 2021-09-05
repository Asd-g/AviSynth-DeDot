#pragma once

#include <cstring>
#include <type_traits>

#include "avisynth.h"

class DeDot : public GenericVideoFilter
{
    int m_luma_2d;
    int m_luma_t;
    int m_chroma_t1;
    int m_chroma_t2;
    bool has_at_least_v8;
    bool process[2];

    void (*proc_chroma)(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);
    void (*proc_luma)(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);

public:
    DeDot(PClip _child, int luma_2d, int luma_t, int chroma_t1, int chroma_t2, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_NICE_FILTER : 0;
    }
};

int process_chroma_pixel_scalar(int pixel_PP, int pixel_P, int pixel_C, int pixel_N, int pixel_NN, int chroma_t1, int chroma_t2);
int process_luma_pixel_scalar(int pixel_current_left, int pixel_current, int pixel_current_right, int pixel_current_2above, int pixel_current_2below,
    int pixel_2previous, int pixel_previous, int pixel_next, int pixel_2next, int luma_2d, int luma_t);

template <typename T>
void process_chroma_plane_sse2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);
template <typename T>
void process_luma_plane_sse2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);

template <typename T>
void process_chroma_plane_avx2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);
template <typename T>
void process_luma_plane_avx2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);
