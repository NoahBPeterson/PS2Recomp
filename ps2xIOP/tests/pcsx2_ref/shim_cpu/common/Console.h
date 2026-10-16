#pragma once
struct ConsoleShim { template <typename... A> void Warning(const char *, A...) {} template <typename... A> void WriteLn(A...) {} };
inline ConsoleShim Console, DevCon;
#define Color_Gray 0
