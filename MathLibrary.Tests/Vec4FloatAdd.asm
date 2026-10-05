.code

Vec4fAdd PROC
    ; RCX = a
    ; RDX = b
    ; R8  = result

    movups xmm0, xmmword ptr [rcx]
    movups xmm1, xmmword ptr [rdx]

    addps  xmm0, xmm1

    movups xmmword ptr [r8], xmm0

    ret
Vec4fAdd ENDP

END
