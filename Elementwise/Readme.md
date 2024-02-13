# Elementwise
Using float2 and float4 to load 64-bit and 128-bit data in a instruction to reduce the number of instructions and store data in L2 cache.
But when we have too much data, L2 cache all miss, which lead to cache thrash and throughput drop down.
Vector load only bring limited performance improment in modern GPU, I don't think it's an important optimization method for now.