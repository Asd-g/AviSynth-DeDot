#include "DeDot.h"
#include "VCL2/vectorclass.h"

template <typename T>
void process_chroma_plane_sse2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2)
{
    const int planes[2] = { PLANAR_U, PLANAR_V };

    const int stride = srcC->GetPitch(PLANAR_U) / sizeof(T);
    const int dst_stride = destf->GetPitch(PLANAR_U) / sizeof(T);
    const int width = srcC->GetRowSize(PLANAR_U) / sizeof(T);
    const int height = srcC->GetHeight(PLANAR_U);

    if constexpr (std::is_same_v<uint8_t, T>)
    {
        const int width_U_simd = width - (width % 16);

        const Vec16uc bytes_chroma_t1 = chroma_t1;
        const Vec16uc bytes_chroma_t2 = chroma_t2;

        for (int i = 0; i < 2; ++i)
        {
            const uint8_t* pPP = srcPP->GetReadPtr(planes[i]);
            const uint8_t* pP = srcP->GetReadPtr(planes[i]);
            const uint8_t* pC = srcC->GetReadPtr(planes[i]);
            const uint8_t* pN = srcN->GetReadPtr(planes[i]);
            const uint8_t* pNN = srcNN->GetReadPtr(planes[i]);
            uint8_t* pD = destf->GetWritePtr(planes[i]);

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width_U_simd; x += 16)
                {
                    const Vec16uc pixel_PP = Vec16uc().load(pPP + x);
                    const Vec16uc pixel_P = Vec16uc().load(pP + x);
                    const Vec16uc pixel_C = Vec16uc().load(pC + x);
                    const Vec16uc pixel_N = Vec16uc().load(pN + x);
                    const Vec16uc pixel_NN = Vec16uc().load(pNN + x);

                    const Vec16uc abs_diff_CP = max(sub_saturated(pixel_C, pixel_P), sub_saturated(pixel_P, pixel_C));
                    const Vec16uc abs_diff_CN = max(sub_saturated(pixel_C, pixel_N), sub_saturated(pixel_N, pixel_C));

                    select(max(sub_saturated(pixel_P, pixel_N), sub_saturated(pixel_N, pixel_P)) <= bytes_chroma_t1 && max(sub_saturated(pixel_C, pixel_PP), sub_saturated(pixel_PP, pixel_C)) <= bytes_chroma_t1 &&
                        max(sub_saturated(pixel_C, pixel_NN), sub_saturated(pixel_NN, pixel_C)) <= bytes_chroma_t1 && abs_diff_CP > bytes_chroma_t2 && abs_diff_CN > bytes_chroma_t2,
                        select(abs_diff_CN <= abs_diff_CP, Vec16uc(_mm_avg_epu8(pixel_N, pixel_C)), Vec16uc(_mm_avg_epu8(pixel_P, pixel_C))), pixel_C).store(pD + x);
                }

                for (int x = width_U_simd; x < width; ++x)
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
    else
    {
        const int width_U_simd = width - (width % 8);

        const Vec8us bytes_chroma_t1 = chroma_t1;
        const Vec8us bytes_chroma_t2 = chroma_t2;

        for (int i = 0; i < 2; ++i)
        {
            const uint16_t* pPP = reinterpret_cast<const uint16_t*>(srcPP->GetReadPtr(planes[i]));
            const uint16_t* pP = reinterpret_cast<const uint16_t*>(srcP->GetReadPtr(planes[i]));
            const uint16_t* pC = reinterpret_cast<const uint16_t*>(srcC->GetReadPtr(planes[i]));
            const uint16_t* pN = reinterpret_cast<const uint16_t*>(srcN->GetReadPtr(planes[i]));
            const uint16_t* pNN = reinterpret_cast<const uint16_t*>(srcNN->GetReadPtr(planes[i]));
            uint16_t* pD = reinterpret_cast<uint16_t*>(destf->GetWritePtr(planes[i]));

            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width_U_simd; x += 8)
                {
                    const Vec8us pixel_PP = Vec8us().load(pPP + x);
                    const Vec8us pixel_P = Vec8us().load(pP + x);
                    const Vec8us pixel_C = Vec8us().load(pC + x);
                    const Vec8us pixel_N = Vec8us().load(pN + x);
                    const Vec8us pixel_NN = Vec8us().load(pNN + x);

                    const Vec8us abs_diff_CP = max(sub_saturated(pixel_C, pixel_P), sub_saturated(pixel_P, pixel_C));
                    const Vec8us abs_diff_CN = max(sub_saturated(pixel_C, pixel_N), sub_saturated(pixel_N, pixel_C));

                    select(max(sub_saturated(pixel_P, pixel_N), sub_saturated(pixel_N, pixel_P)) <= bytes_chroma_t1 && max(sub_saturated(pixel_C, pixel_PP), sub_saturated(pixel_PP, pixel_C)) <= bytes_chroma_t1 &&
                        max(sub_saturated(pixel_C, pixel_NN), sub_saturated(pixel_NN, pixel_C)) <= bytes_chroma_t1 && abs_diff_CP > bytes_chroma_t2 && abs_diff_CN > bytes_chroma_t2,
                        select(abs_diff_CN <= abs_diff_CP, Vec8us(_mm_avg_epu16(pixel_N, pixel_C)), Vec8us(_mm_avg_epu16(pixel_P, pixel_C))), pixel_C).store(pD + x);
                }

                for (int x = width_U_simd; x < width; ++x)
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
}

template void process_chroma_plane_sse2<uint8_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);
template void process_chroma_plane_sse2<uint16_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);

template <class V>
static AVS_FORCEINLINE V temporal_okay(const V pixel_2previous, const V pixel_previous, const V pixel_current, const V pixel_next, const V pixel_2next, const V bytes_luma_t)
{
    if constexpr (std::is_same_v<Vec16uc, V>)
    {
        return select(max(sub_saturated(pixel_previous, pixel_next), sub_saturated(pixel_next, pixel_previous)) <= bytes_luma_t &&
            max(sub_saturated(pixel_current, pixel_2previous), sub_saturated(pixel_2previous, pixel_current)) <= bytes_luma_t &&
            max(sub_saturated(pixel_current, pixel_2next), sub_saturated(pixel_2next, pixel_current)) <= bytes_luma_t,
            select(max(sub_saturated(pixel_next, pixel_current), sub_saturated(pixel_current, pixel_next)) <= max(sub_saturated(pixel_previous, pixel_current), sub_saturated(pixel_current, pixel_previous)),
                Vec16uc(_mm_avg_epu8(pixel_next, pixel_current)), Vec16uc(_mm_avg_epu8(pixel_previous, pixel_current))),
            pixel_current);
    }
    else
    {
        return select(max(sub_saturated(pixel_previous, pixel_next), sub_saturated(pixel_next, pixel_previous)) <= bytes_luma_t &&
            max(sub_saturated(pixel_current, pixel_2previous), sub_saturated(pixel_2previous, pixel_current)) <= bytes_luma_t &&
            max(sub_saturated(pixel_current, pixel_2next), sub_saturated(pixel_2next, pixel_current)) <= bytes_luma_t,
            select(max(sub_saturated(pixel_next, pixel_current), sub_saturated(pixel_current, pixel_next)) <= max(sub_saturated(pixel_previous, pixel_current), sub_saturated(pixel_current, pixel_previous)),
                Vec8us(_mm_avg_epu16(pixel_next, pixel_current)), Vec8us(_mm_avg_epu16(pixel_previous, pixel_current))),
            pixel_current);
    }
}

template <typename T>
void process_luma_plane_sse2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t)
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

    if constexpr (std::is_same_v<uint8_t, T>)
    {
        const int width_simd = (width - 1 * 2) - ((width - 1 * 2) % 16);

        const Vec16uc words_luma_2d = luma_2d;
        const Vec16uc bytes_luma_t = luma_t;

        for (int y = 0; y < 2; ++y)
        {
            memcpy(pD, pC, row_size);

            pD += dst_stride;
            pC += stride;
        }

        for (int y = 2; y < height - 2; ++y)
        {
            pD[0] = pC[0];

            for (int x = 1; x < 1 + width_simd; x += 16)
            {
                // luma2d
                const Vec16uc pixel_current_left = Vec16uc().load(pC + x - 1);
                const Vec16uc pixel_current = Vec16uc().load(pC + x);
                const Vec16uc pixel_current_right = Vec16uc().load(pC + x + 1);

                const Vec16uc pixel_current_2above = Vec16uc().load(pC + x - stride * 2);
                const Vec16uc pixel_current_2below = Vec16uc().load(pC + x + stride * 2);

                const Vec8us left_right_lo = extend_low(pixel_current_left) + extend_low(pixel_current_right);
                const Vec8us left_right_hi = extend_high(pixel_current_left) + extend_high(pixel_current_right);

                const Vec8us above_below_lo = extend_low(pixel_current_2above) + extend_low(pixel_current_2below);
                const Vec8us above_below_hi = extend_high(pixel_current_2above) + extend_high(pixel_current_2below);

                const Vec8us center_center_lo = extend_low(pixel_current) << 1;
                const Vec8us center_center_hi = extend_high(pixel_current) << 1;

                select(Vec16cb(compress_saturated(max(sub_saturated(left_right_lo, center_center_lo), sub_saturated(center_center_lo, left_right_lo)) > extend_low(words_luma_2d),
                    max(sub_saturated(left_right_hi, center_center_hi), sub_saturated(center_center_hi, left_right_hi)) > extend_high(words_luma_2d))) ||
                    Vec16cb(compress_saturated(max(sub_saturated(above_below_lo, center_center_lo), sub_saturated(center_center_lo, above_below_lo)) > extend_low(words_luma_2d),
                        max(sub_saturated(above_below_hi, center_center_hi), sub_saturated(center_center_hi, above_below_hi)) > extend_high(words_luma_2d))),
                    temporal_okay<Vec16uc>(Vec16uc().load(pPP + x), Vec16uc().load(pP + x), pixel_current, Vec16uc().load(pN + x), Vec16uc().load(pNN + x), bytes_luma_t), pixel_current).store(pD + x);
            }

            for (int x = 1 + width_simd; x < width - 1; ++x)
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
    else
    {
        const int width_simd = (width - 1 * 2) - ((width - 1 * 2) % 8);

        const Vec8us words_luma_2d = luma_2d;
        const Vec8us bytes_luma_t = luma_t;

        for (int y = 0; y < 2; ++y)
        {
            memcpy(pD, pC, row_size);

            pD += dst_stride;
            pC += stride;
        }

        for (int y = 2; y < height - 2; ++y)
        {
            pD[0] = pC[0];

            for (int x = 1; x < 1 + width_simd; x += 8)
            {
                // luma2d
                const Vec8us pixel_current_left = Vec8us().load(pC + x - 1);
                const Vec8us pixel_current = Vec8us().load(pC + x);
                const Vec8us pixel_current_right = Vec8us().load(pC + x + 1);

                const Vec8us pixel_current_2above = Vec8us().load(pC + x - stride * 2);
                const Vec8us pixel_current_2below = Vec8us().load(pC + x + stride * 2);

                const Vec4ui left_right_lo = extend_low(pixel_current_left) + extend_low(pixel_current_right);
                const Vec4ui left_right_hi = extend_high(pixel_current_left) + extend_high(pixel_current_right);

                const Vec4ui above_below_lo = extend_low(pixel_current_2above) + extend_low(pixel_current_2below);
                const Vec4ui above_below_hi = extend_high(pixel_current_2above) + extend_high(pixel_current_2below);

                const Vec4ui center_center_lo = extend_low(pixel_current) << 1;
                const Vec4ui center_center_hi = extend_high(pixel_current) << 1;

                select(Vec8sb(compress_saturated(max(sub_saturated(left_right_lo, center_center_lo), sub_saturated(center_center_lo, left_right_lo)) > extend_low(words_luma_2d),
                    max(sub_saturated(left_right_hi, center_center_hi), sub_saturated(center_center_hi, left_right_hi)) > extend_high(words_luma_2d))) ||
                    Vec8sb(compress_saturated(max(sub_saturated(above_below_lo, center_center_lo), sub_saturated(center_center_lo, above_below_lo)) > extend_low(words_luma_2d),
                        max(sub_saturated(above_below_hi, center_center_hi), sub_saturated(center_center_hi, above_below_hi)) > extend_high(words_luma_2d))),
                    temporal_okay<Vec8us>(Vec8us().load(pPP + x), Vec8us().load(pP + x), pixel_current, Vec8us().load(pN + x), Vec8us().load(pNN + x), bytes_luma_t), pixel_current).store(pD + x);
            }

            for (int x = 1 + width_simd; x < width - 1; ++x)
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
}

template void process_luma_plane_sse2<uint8_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);
template void process_luma_plane_sse2<uint16_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);
