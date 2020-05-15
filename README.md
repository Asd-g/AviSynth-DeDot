# Description

Dedot is a temporal cross color (rainbow) and cross luminance (dotcrawl) reduction filter.

The luma and the chroma are filtered completely independently from each other.

It doesn't filter moving objects.

This is [a port of the VapourSynth plugin DeDot](https://github.com/dubhater/vapoursynth-dedot).

# Usage

```
DeDot(clip, int "luma2d", int "lumaT", int "chromaT1", int "chromaT2")
```

## Parameters:
- clip\
    A clip to process. It must have constant format and dimensions and it must be 8 bit Gray or YUV.
    
- luma2d\
    Spatial threshold for the luma. Must be between 0 and 510.\
    Lower values will make the filter process more pixels.\
    If luma_2d is 510, the luma is returned without any processing.\
    Default: 20.

- lumaT\
    Temporal threshold for the luma. Must be between 0 and 255.\
    Higher values will make the filter process more pixels.\
    If luma_t is 0, the luma is returned without any processing.\
    Default: 20.
    
- chromaT1\
    Temporal threshold for the chroma. Must be between 0 and 255.\
    Higher values will make the filter process more pixels.\
    Default: 15.
    
- chromaT2\
    Temporal threshold for the chroma. Must be between 0 and 255.\
    Lower values will make the filter process more pixels.\
    If chroma_t2 is 255, the chroma is returned without any processing.\
    Default: 5.
