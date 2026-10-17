// Stand-in for PCSX2's common/StringUtil.h (Vif.h's desc() helpers).
#pragma once
#include <cstdio>
#include <string>
namespace StringUtil
{
    template <typename... A> std::string StdStringFromFormat(const char *fmt, A... a)
    {
        char buf[128];
        std::snprintf(buf, sizeof(buf), fmt, a...);
        return buf;
    }
}
