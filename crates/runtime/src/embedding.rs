//! CPU reference implementations for embedding lookup.
//!
//! These functions are used for correctness validation against GPU results.
//! They operate on raw tensor bytes read from GGUF files and must NOT
//! depend on Vulkan, tokenization, or any GPU infrastructure.

/// Perform an embedding table lookup on F32 weights.
///
/// GGUF stores the embedding table in column-major order: `[hidden_dim, vocab_size]`.
/// Token `token_id` starts at offset `token_id * hidden_dim` and spans `hidden_dim` elements.
///
/// # Panics
/// Panics if `token_id * hidden_dim + hidden_dim` exceeds `weights.len()`.
pub fn embedding_lookup_f32(weights: &[f32], hidden_dim: usize, token_id: usize) -> Vec<f32> {
    let start = token_id * hidden_dim;
    let end = start + hidden_dim;
    weights[start..end].to_vec()
}

/// Convert a slice of F16 bytes (little-endian pairs) to F32 values.
///
/// Each F16 value is 2 bytes in little-endian order. The returned vector
/// contains `bytes.len() / 2` F32 elements.
///
/// # Panics
/// Panics if `bytes.len()` is not even.
pub fn f16_to_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(2),
        "F16 byte slice must have even length, got {}",
        bytes.len()
    );

    let mut result = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let lo = bytes[i] as u32;
        let hi = bytes[i + 1] as u32;
        let f16_bits = hi << 8 | lo;
        result.push(f16_to_f32(f16_bits));
        i += 2;
    }
    result
}

/// Convert a single IEEE 754 half-precision float (16-bit) to f32.
fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let frac = bits & 0x3FF;

    if exp == 0 {
        if frac == 0 {
            // Zero (positive or negative)
            f32::from_bits(sign << 31)
        } else {
            // Denormal F16 -> normalize to F32
            // Value = frac/1024 * 2^-14
            // Normalize: find leading zeros in 10-bit frac, shift to get implicit 1
            let lz = frac.leading_zeros() - 22;
            let shift = lz + 1;
            let normalized_frac = (frac << shift) & 0x3FF;
            let normalized_exp = 1i32 - shift as i32;
            let e = (normalized_exp - 15 + 127) as u32;
            let m = normalized_frac << 13;
            f32::from_bits((sign << 31) | (e << 23) | m)
        }
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits((sign << 31) | (0xFFu32 << 23) | if frac != 0 { 0x0040_0000 } else { 0 })
    } else {
        // Normal
        let e = (exp - 15 + 127) as u32;
        let m = frac << 13;
        f32::from_bits((sign << 31) | (e << 23) | m)
    }
}

/// Compare two f32 slices element-wise within a tolerance.
///
/// Returns `Ok(())` if all elements match within `tolerance`, or `Err` with
/// the index and values of the first mismatch.
pub fn compare_f32(actual: &[f32], expected: &[f32], tolerance: f32) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "length mismatch: got {} elements, expected {}",
            actual.len(),
            expected.len()
        ));
    }

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > tolerance {
            return Err(format!(
                "mismatch at index {}: got {}, expected {}, diff {}",
                i, a, e, diff
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup_f32_basic() {
        // 3 tokens, hidden_dim=4, column-major layout
        let weights: Vec<f32> = (0..12).map(|x| x as f32).collect();
        // weights = [0,1,2,3, 4,5,6,7, 8,9,10,11]
        // token 0: [0,1,2,3], token 1: [4,5,6,7], token 2: [8,9,10,11]

        assert_eq!(
            embedding_lookup_f32(&weights, 4, 0),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(
            embedding_lookup_f32(&weights, 4, 1),
            vec![4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            embedding_lookup_f32(&weights, 4, 2),
            vec![8.0, 9.0, 10.0, 11.0]
        );
    }

    #[test]
    fn test_embedding_lookup_f32_single_element() {
        let weights = vec![42.0];
        assert_eq!(embedding_lookup_f32(&weights, 1, 0), vec![42.0]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_embedding_lookup_f32_out_of_bounds() {
        let weights = vec![1.0, 2.0];
        embedding_lookup_f32(&weights, 2, 5);
    }

    #[test]
    fn test_f16_to_f32_bytes_basic() {
        // 1.0, 2.0, -1.0, 0.0 in F16 little-endian
        let bytes: [u8; 8] = [0x00, 0x3c, 0x00, 0x40, 0x00, 0xbc, 0x00, 0x00];
        let result = f16_to_f32_bytes(&bytes);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - (-1.0)).abs() < 1e-5);
        assert_eq!(result[3], 0.0);
    }

    #[test]
    fn test_f16_to_f32_bytes_empty() {
        let result = f16_to_f32_bytes(&[]);
        assert!(result.is_empty());
    }

    #[test]
    #[should_panic(expected = "even length")]
    fn test_f16_to_f32_bytes_odd_length() {
        f16_to_f32_bytes(&[0x00]);
    }

    #[test]
    fn test_f16_to_f32_single_zero() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
    }

    #[test]
    fn test_f16_to_f32_single_neg_zero() {
        assert!(f16_to_f32(0x8000).is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_single_one() {
        assert!((f16_to_f32(0x3c00) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_f16_to_f32_single_neg_one() {
        assert!((f16_to_f32(0xbc00) - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_f16_to_f32_single_two() {
        assert!((f16_to_f32(0x4000) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_f16_to_f32_single_inf() {
        assert!(f16_to_f32(0x7c00).is_infinite());
    }

    #[test]
    fn test_f16_to_f32_single_neg_inf() {
        let v = f16_to_f32(0xfc00);
        assert!(v.is_infinite());
        assert!(v.is_sign_negative());
    }

    #[test]
    fn test_f16_to_f32_single_nan() {
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn test_f16_to_f32_small_denormal() {
        // Smallest positive denormal F16
        let v = f16_to_f32(0x0001);
        assert!(v > 0.0);
        assert!((v - 5.9604645e-8).abs() < 1e-10);
    }

    #[test]
    fn test_f16_to_f32_large_normal() {
        // 65504.0 (max F16 normal)
        let v = f16_to_f32(0x7BFF);
        assert!((v - 65504.0).abs() < 0.5, "got {}", v);
    }

    #[test]
    fn test_compare_f32_equal() {
        let a = vec![1.0, 2.0, 3.0];
        let e = vec![1.0, 2.0, 3.0];
        assert!(compare_f32(&a, &e, 1e-5).is_ok());
    }

    #[test]
    fn test_compare_f32_within_tolerance() {
        let a = vec![1.0, 2.0, 3.000001];
        let e = vec![1.0, 2.0, 3.0];
        assert!(compare_f32(&a, &e, 1e-4).is_ok());
    }

    #[test]
    fn test_compare_f32_exceeds_tolerance() {
        let a = vec![1.0, 2.0, 3.1];
        let e = vec![1.0, 2.0, 3.0];
        let err = compare_f32(&a, &e, 1e-5).unwrap_err();
        assert!(err.contains("index 2"));
    }

    #[test]
    fn test_compare_f32_length_mismatch() {
        let a = vec![1.0, 2.0];
        let e = vec![1.0, 2.0, 3.0];
        let err = compare_f32(&a, &e, 1e-5).unwrap_err();
        assert!(err.contains("length mismatch"));
    }

    #[test]
    fn test_compare_f32_empty_slices() {
        assert!(compare_f32(&[], &[], 1e-5).is_ok());
    }
}
