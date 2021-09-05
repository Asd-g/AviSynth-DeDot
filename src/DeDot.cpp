#include <cmath>

#include "DeDot.h"

int process_chroma_pixel_scalar(int pixel_PP, int pixel_P, int pixel_C, int pixel_N, int pixel_NN, int chroma_t1, int chroma_t2)
{
    const int abs_diff_CP = std::abs(pixel_C - pixel_P);
    const int abs_diff_CN = std::abs(pixel_C - pixel_N);

    return (std::abs(pixel_P - pixel_N) <= chroma_t1 &&
        std::abs(pixel_C - pixel_PP) <= chroma_t1 &&
        std::abs(pixel_C - pixel_NN) <= chroma_t1 &&
        abs_diff_CP > chroma_t2 &&
        abs_diff_CN > chroma_t2) ? ((abs_diff_CN <= abs_diff_CP) ? (pixel_N + pixel_C + 1) >> 1 : (pixel_P + pixel_C + 1) >> 1) : pixel_C;
}

int process_luma_pixel_scalar(int pixel_current_left, int pixel_current, int pixel_current_right, int pixel_current_2above, int pixel_current_2below, int pixel_2previous,
    int pixel_previous, int pixel_next, int pixel_2next, int luma_2d, int luma_t)
{
    const int center_center = pixel_current * 2;

    return [&]() {
        if (std::abs(pixel_current_left + pixel_current_right - center_center) > luma_2d || std::abs(pixel_current_2above + pixel_current_2below - center_center) > luma_2d)
        {
            if (std::abs(pixel_previous - pixel_next) <= luma_t &&
                std::abs(pixel_current - pixel_2previous) <= luma_t &&
                std::abs(pixel_current - pixel_2next) <= luma_t)
            {
                return (std::abs(pixel_next - pixel_current) <= std::abs(pixel_previous - pixel_current)) ? (pixel_next + pixel_current + 1) >> 1 : (pixel_previous + pixel_current + 1) >> 1;
            }
            else
                return pixel_current;
        }
        else
            return pixel_current;
    }();
}

template <typename T>
static void process_chroma_plane_scalar(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2)
{
    const int planes[2] = { PLANAR_U, PLANAR_V };

    const int stride = srcC->GetPitch(PLANAR_U) / sizeof(T);
    const int dst_stride = destf->GetPitch(PLANAR_U) / sizeof(T);
    const int width = srcC->GetRowSize(PLANAR_U) / sizeof(T);
    const int height = srcC->GetHeight(PLANAR_U);

    for (int i = 0; i < 2; ++i)
    {
        const T* pPP = reinterpret_cast<const T*>(srcPP->GetReadPtr(planes[i]));
        const T* pP = reinterpret_cast<const T*>(srcP->GetReadPtr(planes[i]));
        const T* pC = reinterpret_cast<const T*>(srcC->GetReadPtr(planes[i]));
        const T* pN = reinterpret_cast<const T*>(srcN->GetReadPtr(planes[i]));
        const T* pNN = reinterpret_cast<const T*>(srcNN->GetReadPtr(planes[i]));
        T* pD = reinterpret_cast<T*>(destf->GetWritePtr(planes[i]));

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
                pD[x] = process_chroma_pixel_scalar(pPP[x], pP[x], pC[x], pN[x], pNN[x], chroma_t1, chroma_t2);

            pPP += stride;
            pP += stride;
            pC += stride;
            pN += stride;
            pNN += stride;
            pD += dst_stride;
        }
    }
}

template <typename T>
static void process_luma_plane_scalar(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t)
{
    const int stride = srcC->GetPitch() / sizeof(T);
    const int dst_stride = destf->GetPitch() / sizeof(T);
    const int row_size = srcC->GetRowSize();
    const int width = row_size / sizeof(T);
    const int height = srcC->GetHeight();
    const T* pPP = reinterpret_cast<const T*>(srcPP->GetReadPtr()) + static_cast<int64_t>(2) * stride;
    const T* pP = reinterpret_cast<const T*>(srcP->GetReadPtr()) + static_cast<int64_t>(2) * stride;
    const T* pC = reinterpret_cast<const T*>(srcC->GetReadPtr());
    const T* pN = reinterpret_cast<const T*>(srcN->GetReadPtr()) + static_cast<int64_t>(2) * stride;
    const T* pNN = reinterpret_cast<const T*>(srcNN->GetReadPtr()) + static_cast<int64_t>(2) * stride;
    T* pD = reinterpret_cast<T*>(destf->GetWritePtr());

    for (int y = 0; y < 2; ++y)
    {
        memcpy(pD, pC, row_size);

        pD += dst_stride;
        pC += stride;
    }

    for (int y = 2; y < height - 2; ++y)
    {
        pD[0] = pC[0];

        for (int x = 1; x < width - 1; ++x)
            pD[x] = process_luma_pixel_scalar(pC[x - 1], pC[x], pC[x + 1], pC[x - stride * 2], pC[x + stride * 2],
                pPP[x], pP[x], pN[x], pNN[x], luma_2d, luma_t);

        pD[width - 1] = pC[width - 1];

        pPP += stride;
        pP += stride;
        pC += stride;
        pN += stride;
        pNN += stride;
        pD += dst_stride;
    }

    for (int y = height - 2; y < height; ++y)
    {
        memcpy(pD, pC, row_size);

        pD += dst_stride;
        pC += stride;
    }
}

DeDot::DeDot(PClip _child, int luma_2d, int luma_t, int chroma_t1, int chroma_t2, int opt, IScriptEnvironment* env)
    : GenericVideoFilter(_child), m_luma_2d(luma_2d), m_luma_t(luma_t), m_chroma_t1(chroma_t1), m_chroma_t2(chroma_t2)
{
    const int comp_size = vi.ComponentSize();

    if (comp_size > 2 || vi.IsRGB() || !vi.IsPlanar())
        env->ThrowError("DeDot: the input clip must be in 8..16-bit YUV or Gray planar format.");
    if (m_luma_2d < 0 || m_luma_2d > 510)
        env->ThrowError("DeDot: luma2d must be between 0 and 510 (inclusive).");
    if (m_luma_t < 0 || m_luma_t > 255)
        env->ThrowError("DeDot: lumaT must be between 0 and 255 (inclusive).");
    if (m_chroma_t1 < 0 || m_chroma_t1 > 255)
        env->ThrowError("DeDot: chromaT1 must be between 0 and 255 (inclusive).");
    if (m_chroma_t2 < 0 || m_chroma_t2 > 255)
        env->ThrowError("DeDot: chromaT2 must be between 0 and 255 (inclusive).");
    if ((m_luma_2d == 510 || m_luma_t == 0) && m_chroma_t2 == 255)
        env->ThrowError("DeDot: chromaT2 can't be 255 when luma2d is 510 or when lumaT is 0 because then all the planes would be returned unchanged.");
    if (opt < -1 || opt > 2)
        env->ThrowError("DeDot: opt must be between -1 and 2 (inclusive).");
    if (opt == 1 && !(env->GetCPUFlags() & CPUF_SSE2))
        env->ThrowError("DeDot: opt=1 requires SSE2.");
    if (opt == 2 && !(env->GetCPUFlags() & CPUF_AVX2))
        env->ThrowError("DeDot: opt=1 requires SSE2.");

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; }

    process[0] = m_luma_2d < 510 && m_luma_t > 0;
    process[1] = m_chroma_t2 < 255 && vi.NumComponents() > 1;

    if (comp_size == 2)
    {
        const int peak = (1 << vi.BitsPerComponent()) - 1;
        m_luma_2d *= peak / 255;
        m_luma_t *= peak / 255;
        m_chroma_t1 *= peak / 255;
        m_chroma_t2 *= peak / 255;

        if ((opt < 0 && !!(env->GetCPUFlags() & CPUF_AVX2)) || opt == 2)
        {
            proc_chroma = (process[1]) ? process_chroma_plane_avx2<uint16_t> : nullptr;
            proc_luma = (process[0]) ? process_luma_plane_avx2<uint16_t> : nullptr;
        }
        else if ((opt < 0 && !!(env->GetCPUFlags() & CPUF_SSE2)) || opt == 1)
        {
            proc_chroma = (process[1]) ? process_chroma_plane_sse2<uint16_t> : nullptr;
            proc_luma = (process[0]) ? process_luma_plane_sse2<uint16_t> : nullptr;
        }
        else
        {
            proc_chroma = (process[1]) ? process_chroma_plane_scalar<uint16_t> : nullptr;
            proc_luma = (process[0]) ? process_luma_plane_scalar<uint16_t> : nullptr;
        }
    }
    else
    {
        if ((opt < 0 && !!(env->GetCPUFlags() & CPUF_AVX2)) || opt == 2)
        {
            proc_chroma = (process[1]) ? process_chroma_plane_avx2<uint8_t> : nullptr;
            proc_luma = (process[0]) ? process_luma_plane_avx2<uint8_t> : nullptr;
        }
        else if ((opt < 0 && !!(env->GetCPUFlags() & CPUF_SSE2)) || opt == 1)
        {
            proc_chroma = (process[1]) ? process_chroma_plane_sse2<uint8_t> : nullptr;
            proc_luma = (process[0]) ? process_luma_plane_sse2<uint8_t> : nullptr;
        }
        else
        {
            proc_chroma = (process[1]) ? process_chroma_plane_scalar<uint8_t> : nullptr;
            proc_luma = (process[0]) ? process_luma_plane_scalar<uint8_t> : nullptr;
        }
    }
};

PVideoFrame __stdcall DeDot::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame srcPP = child->GetFrame(n - 2, env);
    PVideoFrame srcP = child->GetFrame(n - 1, env);
    PVideoFrame srcC = child->GetFrame(n, env);
    PVideoFrame srcN = child->GetFrame(n + 1, env);
    PVideoFrame srcNN = child->GetFrame(n + 2, env);
    PVideoFrame destf = (has_at_least_v8) ? env->NewVideoFrameP(vi, &srcC) : env->NewVideoFrame(vi);
    
    if (process[1])
        proc_chroma(srcPP, srcP, srcC, srcN, srcNN, destf, m_chroma_t1, m_chroma_t2);
    else
    {
        const int planes[2] = { PLANAR_U, PLANAR_V };
        const int* plane = planes;

        for (int i = 0; i < 2; ++i)
            env->BitBlt(destf->GetWritePtr(plane[i]), destf->GetPitch(plane[i]), srcC->GetReadPtr(planes[i]), srcC->GetPitch(plane[i]), srcC->GetRowSize(plane[i]), srcC->GetHeight(plane[i]));
    }

    if (process[0])
        proc_luma(srcPP, srcP, srcC, srcN, srcNN, destf, m_luma_2d, m_luma_t);
    else
        env->BitBlt(destf->GetWritePtr(), destf->GetPitch(), srcC->GetReadPtr(), srcC->GetPitch(), srcC->GetRowSize(), srcC->GetHeight());

    return destf;
};

AVSValue __cdecl Create_DeDot(AVSValue args, void *user_data, IScriptEnvironment *env)
{	
    return new DeDot(
        args[0].AsClip(),
        args[1].AsInt(20),
        args[2].AsInt(20),
        args[3].AsInt(15),
        args[4].AsInt(5),
        args[5].AsInt(-1),
        env);
}

const AVS_Linkage *AVS_linkage;

extern "C" __declspec(dllexport)
const char * __stdcall AvisynthPluginInit3(IScriptEnvironment *env, const AVS_Linkage *const vectors)
{
    AVS_linkage = vectors;

    env->AddFunction("DeDot", "c[luma2d]i[lumaT]i[chromaT1]i[chromaT2]i[opt]i", Create_DeDot, 0);
    return "DeDot";
}
