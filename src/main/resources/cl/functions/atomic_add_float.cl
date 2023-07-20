// inline void float_atomic_add(__global float *loc, const float f)
// {
//     private float old = *loc;
//     private float sum = old + f;
//     while(atomic_cmpxchg((__global int*)loc, *((int*)&old), *((int*)&sum)) != *((int*)&old))
//     {
//         old = *loc;
//         sum = old + f;
//     }
// }
// static float atomic_cmpxchg_f32(volatile __global float *p, float cmp, float val) 
// {
// 	union 
//     {
// 		unsigned int u32;
// 		float        f32;
// 	} cmp_union, val_union, old_union;

// 	cmp_union.f32 = cmp;
// 	val_union.f32 = val;
// 	old_union.u32 = atomic_cmpxchg((volatile __global unsigned int *) p, cmp_union.u32, val_union.u32);
// 	return old_union.f32;
// }

// static float atomic_add_f32(volatile __global float *p, float val) 
// {
// 	float found = *p;
// 	float expected;
// 	do 
//     {
// 		expected = found;
// 		found = atomic_cmpxchg_f32(p, expected, expected + val);
// 	} while (found != expected);
	
//     return found;
// }