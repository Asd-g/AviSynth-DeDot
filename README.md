## Description

Dedot is a temporal cross color (rainbow) and cross luminance (dotcrawl) reduction filter.

The luma and the chroma are filtered completely independently from each other.

It doesn't filter moving objects.

This is [a port of the VapourSynth plugin DeDot](https://github.com/dubhater/vapoursynth-dedot).

### Requirements:

- AviSynth 2.60 / AviSynth+ 3.4 or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases)) (Windows only)

### Usage:

```
DeDot(clip, int "luma2d", int "lumaT", int "chromaT1", int "chromaT2", int "opt")
```

### Parameters:
- clip\
    A clip to process. It must be in YUV 8..16-bit planar format.
    
- luma2d\
    Spatial threshold for the luma.\
    Must be between 0 and 510.\
    Lower values will make the filter process more pixels.\
    If luma_2d is 510, the luma is returned without any processing.\
    Default: 20.

- lumaT\
    Temporal threshold for the luma.\
    Must be between 0 and 255.\
    Higher values will make the filter process more pixels.\
    If luma_t is 0, the luma is returned without any processing.\
    Default: 20.
    
- chromaT1\
    Temporal threshold for the chroma.\
    Must be between 0 and 255.\
    Higher values will make the filter process more pixels.\
    Default: 15.
    
- chromaT2\
    Temporal threshold for the chroma.\
    Must be between 0 and 255.\
    Lower values will make the filter process more pixels.\
    If chroma_t2 is 255, the chroma is returned without any processing.\
    Default: 5.

- opt\
    Sets which cpu optimizations to use.\
    -1: Auto-detect.\
    0: Use C++ code.\
    1: Use SSE2 code.\
    2: Use AVX2 code.\
    Default: -1.
    
### Building:

- Windows\
    Use solution files.

- Linux
    ```
    Requirements:
        - Git
        - C++17 compiler
        - CMake >= 3.16
    ```
    ```
    git clone https://github.com/Asd-g/AviSynth-DeDot && \
    cd AviSynth-DeDot && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    sudo make install
    ```
