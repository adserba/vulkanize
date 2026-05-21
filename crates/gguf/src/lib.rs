use std::fmt;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const HEADER_SIZE: usize = 24;
const SUPPORTED_VERSIONS: [u32; 1] = [3];

pub const GGUF_MAGIC_STR: &str = "GGUF";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug)]
pub enum GgufError {
    FileTooShort { required: usize, available: usize },
    InvalidMagic { got: [u8; 4] },
    UnsupportedVersion { version: u32 },
    Io(std::io::Error),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::FileTooShort { required, available } => write!(
                f,
                "file too short: need {} bytes, got {}",
                required, available
            ),
            GgufError::InvalidMagic { got } => write!(
                f,
                "invalid magic: expected {}, got {:?}",
                GGUF_MAGIC_STR,
                std::str::from_utf8(got).unwrap_or("<invalid utf8>")
            ),
            GgufError::UnsupportedVersion { version } => write!(
                f,
                "unsupported GGUF version {}: supported versions are {:?}",
                version, SUPPORTED_VERSIONS
            ),
            GgufError::Io(e) => write!(f, "io error: {}", e),
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// Parse a GGUF header from a byte slice (at least 28 bytes).
pub fn parse_header(bytes: &[u8]) -> Result<GgufHeader, GgufError> {
    if bytes.len() < HEADER_SIZE {
        return Err(GgufError::FileTooShort {
            required: HEADER_SIZE,
            available: bytes.len(),
        });
    }

    let magic = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != GGUF_MAGIC {
        return Err(GgufError::InvalidMagic { got: magic });
    }

    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let tensor_count = u64::from_le_bytes([
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    ]);
    let metadata_kv_count = u64::from_le_bytes([
        bytes[16], bytes[17], bytes[18], bytes[19], bytes[20], bytes[21], bytes[22], bytes[23],
    ]);

    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(GgufError::UnsupportedVersion { version });
    }

    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Parse a GGUF header from a file on disk.
pub fn parse_header_from_path<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<GgufHeader, GgufError> {
    let bytes = std::fs::read(path)?;
    parse_header(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(magic: [u8; 4], version: u32, tensor_count: u64, kv_count: u64) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE);
        buf.extend_from_slice(&magic);
        buf.extend_from_slice(&version.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf
    }

    #[test]
    fn test_parse_valid_header() {
        let bytes = make_header(*b"GGUF", 3, 42, 17);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 42);
        assert_eq!(header.metadata_kv_count, 17);
    }

    #[test]
    fn test_parse_zero_counts() {
        let bytes = make_header(*b"GGUF", 3, 0, 0);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.tensor_count, 0);
        assert_eq!(header.metadata_kv_count, 0);
    }

    #[test]
    fn test_parse_large_counts() {
        let bytes = make_header(*b"GGUF", 3, u64::MAX, u64::MAX);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.tensor_count, u64::MAX);
        assert_eq!(header.metadata_kv_count, u64::MAX);
    }

    #[test]
    fn test_file_too_short_exact() {
        let bytes = vec![0u8; HEADER_SIZE - 1];
        match parse_header(&bytes) {
            Err(GgufError::FileTooShort { required, available }) => {
                assert_eq!(required, HEADER_SIZE);
                assert_eq!(available, HEADER_SIZE - 1);
            }
            other => panic!("expected FileTooShort, got {:?}", other),
        }
    }

    #[test]
    fn test_file_too_short_empty() {
        match parse_header(&[]) {
            Err(GgufError::FileTooShort { required, available }) => {
                assert_eq!(required, HEADER_SIZE);
                assert_eq!(available, 0);
            }
            other => panic!("expected FileTooShort, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_magic() {
        let bytes = make_header(*b"gguf", 3, 0, 0);
        match parse_header(&bytes) {
            Err(GgufError::InvalidMagic { got }) => {
                assert_eq!(got, *b"gguf");
            }
            other => panic!("expected InvalidMagic, got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_magic_random() {
        let bytes = make_header(*b"XXXX", 3, 0, 0);
        match parse_header(&bytes) {
            Err(GgufError::InvalidMagic { .. }) => {}
            other => panic!("expected InvalidMagic, got {:?}", other),
        }
    }

    #[test]
    fn test_unsupported_version_v2() {
        let bytes = make_header(*b"GGUF", 2, 0, 0);
        match parse_header(&bytes) {
            Err(GgufError::UnsupportedVersion { version }) => {
                assert_eq!(version, 2);
            }
            other => panic!("expected UnsupportedVersion, got {:?}", other),
        }
    }

    #[test]
    fn test_unsupported_version_v4() {
        let bytes = make_header(*b"GGUF", 4, 0, 0);
        match parse_header(&bytes) {
            Err(GgufError::UnsupportedVersion { version }) => {
                assert_eq!(version, 4);
            }
            other => panic!("expected UnsupportedVersion, got {:?}", other),
        }
    }

    #[test]
    fn test_unsupported_version_zero() {
        let bytes = make_header(*b"GGUF", 0, 0, 0);
        assert!(matches!(
            parse_header(&bytes),
            Err(GgufError::UnsupportedVersion { version: 0 })
        ));
    }

    #[test]
    fn test_error_display_file_too_short() {
        let err = GgufError::FileTooShort {
            required: 24,
            available: 10,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("file too short"));
        assert!(msg.contains("24"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn test_error_display_invalid_magic() {
        let err = GgufError::InvalidMagic { got: *b"XXXX" };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid magic"));
        assert!(msg.contains("GGUF"));
    }

    #[test]
    fn test_error_display_unsupported_version() {
        let err = GgufError::UnsupportedVersion { version: 2 };
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported"));
        assert!(msg.contains("2"));
    }

    #[test]
    fn test_header_is_copy() {
        let bytes = make_header(*b"GGUF", 3, 1, 2);
        let h1 = parse_header(&bytes).unwrap();
        let h2 = h1;
        assert_eq!(h1.version, h2.version);
        assert_eq!(h1.tensor_count, h2.tensor_count);
    }

    #[test]
    fn test_parse_header_exact_size() {
        let bytes = make_header(*b"GGUF", 3, 5, 3);
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 5);
        assert_eq!(header.metadata_kv_count, 3);
    }

    #[test]
    fn test_parse_header_with_trailing_data() {
        let mut bytes = make_header(*b"GGUF", 3, 5, 3);
        bytes.extend_from_slice(b"extra data that should be ignored");
        let header = parse_header(&bytes).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 5);
        assert_eq!(header.metadata_kv_count, 3);
    }
}
