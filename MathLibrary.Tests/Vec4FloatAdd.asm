.code

Vec4fAdd PROC
    ; RCX = a
    ; RDX = b
    ; R8  = number of Vec4f
    ; R9: counts vectors remaining; exits at 0 (no vectors left)

    mov r9, r8
    shr r9, 1

    test r9, r9
    jz remainder

add_loop:
    vmovups ymm0, ymmword ptr [rcx]
    vmovups ymm1, ymmword ptr [rdx]

    vaddps ymm0, ymm0, ymm1 ; ymm0 += ymm1

    vmovups ymmword ptr [rcx], ymm0

    add rcx, 32 ; 4 floats increment, moves onto the next vector
    add rdx, 32 ; same
    dec r9
    jnz add_loop

remainder:
    test r8, 1
    jz done

    movups xmm0, xmmword ptr [rcx]
    movups xmm1, xmmword ptr [rdx]

    addps xmm0, xmm1

    movups xmmword ptr [rcx], xmm0

done:
    vzeroupper ; Reset registers
    ret

Vec4fAdd ENDP

END
