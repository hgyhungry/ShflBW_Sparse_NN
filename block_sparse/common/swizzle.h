#pragma once
#include "base.h"

struct SwizzleIdentity {
    DEVICE_INLINE
    int operator()(int offset) {
        return offset;
    }
};

struct Swizzle8BWiseXor {
    DEVICE_INLINE
    int operator()(int offset) {
        return (offset ^ 
                ((offset & (7<<6))>>3));
    }
};