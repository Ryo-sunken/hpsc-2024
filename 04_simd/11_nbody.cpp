#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], j[N];
  for (int i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    j[i] = i;
  }
  __m512 jvec = _mm512_load_ps(j);
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  for (int i = 0; i < N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 ivec = _mm512_set1_ps(i);
    __mmask16 mask = _mm512_cmp_ps_mask(ivec, jvec, _MM_CMPINT_NE);

    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);
    __m512 r = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry)));
    __m512 dfx = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(rx, mvec), r), r), r));
    __m512 dfy = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(ry, mvec), r), r), r));
    fx[i] -= _mm512_reduce_add_ps(dfx);
    fy[i] -= _mm512_reduce_add_ps(dfy);
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}