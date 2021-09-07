#include "DeDot.h"
#include "VCL2/vectorclass.h"

template <typename T>
void process_chroma_plane_avx2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2)
{
    const int planes[2] = { PLANAR_U, PLANAR_V };

    const int stride = srcC->GetPitch(PLANAR_U) / sizeof(T);
    const int dst_stride = destf->GetPitch(PLANAR_U) / sizeof(T);
    const int width = srcC->GetRowSize(PLANAR_U) / sizeof(T);
    const int height = srcC->GetHeight(PLANAR_U);

    if constexpr (std::is_same_v<uint8_t, T>)
    {
        const int width_U_simd = width - (width % 32);

        const Vec32uc bytes_chroma_t1 = chroma_t1;
        const Vec32uc bytes_chroma_t2 = chroma_t2;

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
                for (int x = 0; x < width_U_simd; x += 32)
                {
                    const Vec32uc pixel_PP = Vec32uc().load(pPP + x);
                    const Vec32uc pixel_P = Vec32uc().load(pP + x);
                    const Vec32uc pixel_C = Vec32uc().load(pC + x);
                    const Vec32uc pixel_N = Vec32uc().load(pN + x);
                    const Vec32uc pixel_NN = Vec32uc().load(pNN + x);

                    const Vec32uc abs_diff_CP = max(sub_saturated(pixel_C, pixel_P), sub_saturated(pixel_P, pixel_C));
                    const Vec32uc abs_diff_CN = max(sub_saturated(pixel_C, pixel_N), sub_saturated(pixel_N, pixel_C));

                    select(max(sub_saturated(pixel_P, pixel_N), sub_saturated(pixel_N, pixel_P)) <= bytes_chroma_t1 && max(sub_saturated(pixel_C, pixel_PP), sub_saturated(pixel_PP, pixel_C)) <= bytes_chroma_t1 &&
                        max(sub_saturated(pixel_C, pixel_NN), sub_saturated(pixel_NN, pixel_C)) <= bytes_chroma_t1 && abs_diff_CP > bytes_chroma_t2 && abs_diff_CN > bytes_chroma_t2,
                        select(abs_diff_CN <= abs_diff_CP, Vec32uc(_mm256_avg_epu8(pixel_N, pixel_C)), Vec32uc(_mm256_avg_epu8(pixel_P, pixel_C))), pixel_C).store(pD + x);
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
        const int width_U_simd = width - (width % 16);

        const Vec16us bytes_chroma_t1 = chroma_t1;
        const Vec16us bytes_chroma_t2 = chroma_t2;

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
                for (int x = 0; x < width_U_simd; x += 16)
                {
                    const Vec16us pixel_PP = Vec16us().load(pPP + x);
                    const Vec16us pixel_P = Vec16us().load(pP + x);
                    const Vec16us pixel_C = Vec16us().load(pC + x);
                    const Vec16us pixel_N = Vec16us().load(pN + x);
                    const Vec16us pixel_NN = Vec16us().load(pNN + x);

                    const Vec16us abs_diff_CP = max(sub_saturated(pixel_C, pixel_P), sub_saturated(pixel_P, pixel_C));
                    const Vec16us abs_diff_CN = max(sub_saturated(pixel_C, pixel_N), sub_saturated(pixel_N, pixel_C));

                    select(max(sub_saturated(pixel_P, pixel_N), sub_saturated(pixel_N, pixel_P)) <= bytes_chroma_t1 && max(sub_saturated(pixel_C, pixel_PP), sub_saturated(pixel_PP, pixel_C)) <= bytes_chroma_t1 &&
                        max(sub_saturated(pixel_C, pixel_NN), sub_saturated(pixel_NN, pixel_C)) <= bytes_chroma_t1 && abs_diff_CP > bytes_chroma_t2 && abs_diff_CN > bytes_chroma_t2,
                        select(abs_diff_CN <= abs_diff_CP, Vec16us(_mm256_avg_epu16(pixel_N, pixel_C)), Vec16us(_mm256_avg_epu16(pixel_P, pixel_C))), pixel_C).store(pD + x);
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

template void process_chroma_plane_avx2<uint8_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);
template void process_chroma_plane_avx2<uint16_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int chroma_t1, const int chroma_t2);

template <typename T>
void process_luma_plane_avx2(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t)
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
        const int width_simd = (width - 1 * 2) - ((width - 1 * 2) % 32);

        const Vec32uc words_luma_2d = luma_2d;
        const Vec32uc bytes_luma_t = luma_t;

        for (int y = 0; y < 2; ++y)
        {
            memcpy(pD, pC, row_size);

            pD += dst_stride;
            pC += stride;
        }

        for (int y = 2; y < height - 2; ++y)
        {
            pD[0] = pC[0];

            for (int x = 1; x < 1 + width_simd; x += 32)
            {
                // luma2d
                const Vec32uc pixel_current_left = Vec32uc().load(pC + x - 1);
                const Vec32uc pixel_current = Vec32uc().load(pC + x);
                const Vec32uc pixel_current_right = Vec32uc().load(pC + x + 1);

                const Vec32uc pixel_current_2above = Vec32uc().load(pC + x - stride * 2);
                const Vec32uc pixel_current_2below = Vec32uc().load(pC + x + stride * 2);

                const Vec16us left_right_lo = extend_low(pixel_current_left) + extend_low(pixel_current_right);
                const Vec16us left_right_hi = extend_high(pixel_current_left) + extend_high(pixel_current_right);

                const Vec16us above_below_lo = extend_low(pixel_current_2above) + extend_low(pixel_current_2below);
                const Vec16us above_below_hi = extend_high(pixel_current_2above) + extend_high(pixel_current_2below);

                const Vec16us center_center_lo = extend_low(pixel_current) << 1;
                const Vec16us center_center_hi = extend_high(pixel_current) << 1;

                Vec32c spat_check = compress_saturated(max(sub_saturated(left_right_lo, center_center_lo), sub_saturated(center_center_lo, left_right_lo)) > extend_low(words_luma_2d),
                    max(sub_saturated(left_right_hi, center_center_hi), sub_saturated(center_center_hi, left_right_hi)) > extend_high(words_luma_2d)) ||
                    compress_saturated(max(sub_saturated(above_below_lo, center_center_lo), sub_saturated(center_center_lo, above_below_lo)) > extend_low(words_luma_2d),
                        max(sub_saturated(above_below_hi, center_center_hi), sub_saturated(center_center_hi, above_below_hi)) > extend_high(words_luma_2d));

                int64_t all_pixels_lo, all_pixels_hi;
                const Vec16c sp_lo = spat_check.get_low();
                const Vec16c sp_hi = spat_check.get_high();
                compress_saturated(Vec8s(sp_lo), Vec8s(sp_lo)).storel(&all_pixels_lo);
                compress_saturated(Vec8s(sp_hi), Vec8s(sp_hi)).storel(&all_pixels_hi);

                if (all_pixels_lo || all_pixels_hi)
                {
                    const Vec32uc pixel_2previous = Vec32uc().load(pPP + x);
                    const Vec32uc pixel_previous = Vec32uc().load(pP + x);
                    const Vec32uc pixel_next = Vec32uc().load(pN + x);
                    const Vec32uc pixel_2next = Vec32uc().load(pNN + x);

                    Vec32c temp_check = max(sub_saturated(pixel_previous, pixel_next), sub_saturated(pixel_next, pixel_previous)) <= bytes_luma_t &&
                        max(sub_saturated(pixel_current, pixel_2previous), sub_saturated(pixel_2previous, pixel_current)) <= bytes_luma_t &&
                        max(sub_saturated(pixel_current, pixel_2next), sub_saturated(pixel_2next, pixel_current)) <= bytes_luma_t;

                    int64_t temp_lo, temp_hi;
                    const Vec16c tm_lo = temp_check.get_low();
                    const Vec16c tm_hi = temp_check.get_high();
                    compress_saturated(Vec8s(tm_lo), Vec8s(tm_lo)).storel(&temp_lo);
                    compress_saturated(Vec8s(tm_hi), Vec8s(tm_hi)).storel(&temp_hi);

                    if (temp_lo || temp_hi)
                    {
                        select(Vec32cb(spat_check),
                            select(Vec32cb(temp_check),
                                select(max(sub_saturated(pixel_next, pixel_current), sub_saturated(pixel_current, pixel_next)) <= max(sub_saturated(pixel_previous, pixel_current), sub_saturated(pixel_current, pixel_previous)),
                                    Vec32uc(_mm256_avg_epu8(pixel_next, pixel_current)), Vec32uc(_mm256_avg_epu8(pixel_previous, pixel_current))),
                                pixel_current),
                            pixel_current).store(pD + x);
                    }
                    else
                        pixel_current.store(pD + x);
                }
                else
                    pixel_current.store(pD + x);
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
        const int width_simd = (width - 1 * 2) - ((width - 1 * 2) % 16);

        const Vec16us words_luma_2d = luma_2d;
        const Vec16us bytes_luma_t = luma_t;

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
                const Vec16us pixel_current_left = Vec16us().load(pC + x - 1);
                const Vec16us pixel_current = Vec16us().load(pC + x);
                const Vec16us pixel_current_right = Vec16us().load(pC + x + 1);

                const Vec16us pixel_current_2above = Vec16us().load(pC + x - stride * 2);
                const Vec16us pixel_current_2below = Vec16us().load(pC + x + stride * 2);

                const Vec8ui left_right_lo = extend_low(pixel_current_left) + extend_low(pixel_current_right);
                const Vec8ui left_right_hi = extend_high(pixel_current_left) + extend_high(pixel_current_right);

                const Vec8ui above_below_lo = extend_low(pixel_current_2above) + extend_low(pixel_current_2below);
                const Vec8ui above_below_hi = extend_high(pixel_current_2above) + extend_high(pixel_current_2below);

                const Vec8ui center_center_lo = extend_low(pixel_current) << 1;
                const Vec8ui center_center_hi = extend_high(pixel_current) << 1;

                Vec16s spat_check = compress_saturated(max(sub_saturated(left_right_lo, center_center_lo), sub_saturated(center_center_lo, left_right_lo)) > extend_low(words_luma_2d),
                    max(sub_saturated(left_right_hi, center_center_hi), sub_saturated(center_center_hi, left_right_hi)) > extend_high(words_luma_2d)) ||
                    compress_saturated(max(sub_saturated(above_below_lo, center_center_lo), sub_saturated(center_center_lo, above_below_lo)) > extend_low(words_luma_2d),
                        max(sub_saturated(above_below_hi, center_center_hi), sub_saturated(center_center_hi, above_below_hi)) > extend_high(words_luma_2d));

                int64_t all_pixels_lo, all_pixels_hi;
                const Vec8s sp_lo = spat_check.get_low();
                const Vec8s sp_hi = spat_check.get_high();
                compress_saturated(Vec4i(sp_lo), Vec4i(sp_lo)).storel(&all_pixels_lo);
                compress_saturated(Vec4i(sp_hi), Vec4i(sp_hi)).storel(&all_pixels_hi);

                if (all_pixels_lo || all_pixels_hi)
                {
                    const Vec16us pixel_2previous = Vec16us().load(pPP + x);
                    const Vec16us pixel_previous = Vec16us().load(pP + x);
                    const Vec16us pixel_next = Vec16us().load(pN + x);
                    const Vec16us pixel_2next = Vec16us().load(pNN + x);

                    Vec16s temp_check = max(sub_saturated(pixel_previous, pixel_next), sub_saturated(pixel_next, pixel_previous)) <= bytes_luma_t &&
                        max(sub_saturated(pixel_current, pixel_2previous), sub_saturated(pixel_2previous, pixel_current)) <= bytes_luma_t &&
                        max(sub_saturated(pixel_current, pixel_2next), sub_saturated(pixel_2next, pixel_current)) <= bytes_luma_t;

                    int64_t temp_lo, temp_hi;
                    const Vec8s tm_lo = temp_check.get_low();
                    const Vec8s tm_hi = temp_check.get_high();
                    compress_saturated(Vec4i(tm_lo), Vec4i(tm_lo)).storel(&temp_lo);
                    compress_saturated(Vec4i(tm_hi), Vec4i(tm_hi)).storel(&temp_hi);

                    if (temp_lo || temp_hi)
                    {
                        select(Vec16sb(spat_check),
                            select(Vec16sb(temp_check),
                                select(max(sub_saturated(pixel_next, pixel_current), sub_saturated(pixel_current, pixel_next)) <= max(sub_saturated(pixel_previous, pixel_current), sub_saturated(pixel_current, pixel_previous)),
                                    Vec16us(_mm256_avg_epu16(pixel_next, pixel_current)), Vec16us(_mm256_avg_epu16(pixel_previous, pixel_current))),
                                pixel_current),
                            pixel_current).store(pD + x);
                    }
                    else
                        pixel_current.store(pD + x);
                }
                else
                    pixel_current.store(pD + x);
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

template void process_luma_plane_avx2<uint8_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);
template void process_luma_plane_avx2<uint16_t>(PVideoFrame& srcPP, PVideoFrame& srcP, PVideoFrame& srcC, PVideoFrame& srcN, PVideoFrame& srcNN, PVideoFrame& destf, const int luma_2d, const int luma_t);
