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
    TruncatedMetadata { context: &'static str },
    TruncatedTensor { context: &'static str },
    InvalidUtf8,
    UnsupportedValueType { type_id: u32 },
    UnsupportedTensorType { type_id: u32 },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufError::FileTooShort {
                required,
                available,
            } => write!(
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
            GgufError::TruncatedMetadata { context } => {
                write!(f, "truncated metadata while reading: {}", context)
            }
            GgufError::InvalidUtf8 => write!(f, "invalid UTF-8 in metadata string"),
            GgufError::UnsupportedValueType { type_id } => {
                write!(f, "unsupported metadata value type: {}", type_id)
            }
            GgufError::TruncatedTensor { context } => {
                write!(f, "truncated tensor header while reading: {}", context)
            }
            GgufError::UnsupportedTensorType { type_id } => {
                write!(f, "unsupported tensor data type: {}", type_id)
            }
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(e: std::io::Error) -> Self {
        GgufError::Io(e)
    }
}

/// GGUF metadata value type IDs (from llama.cpp gguf.h)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    pub fn to_u32(self) -> u32 {
        self as u32
    }

    pub fn from_u32(id: u32) -> Option<Self> {
        match id {
            0 => Some(GgufValueType::Uint8),
            1 => Some(GgufValueType::Int8),
            2 => Some(GgufValueType::Uint16),
            3 => Some(GgufValueType::Int16),
            4 => Some(GgufValueType::Uint32),
            5 => Some(GgufValueType::Int32),
            6 => Some(GgufValueType::Float32),
            7 => Some(GgufValueType::Bool),
            8 => Some(GgufValueType::String),
            9 => Some(GgufValueType::Array),
            10 => Some(GgufValueType::Uint64),
            11 => Some(GgufValueType::Int64),
            12 => Some(GgufValueType::Float64),
            _ => None,
        }
    }
}

impl fmt::Display for GgufValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufValueType::Uint8 => write!(f, "uint8"),
            GgufValueType::Int8 => write!(f, "int8"),
            GgufValueType::Uint16 => write!(f, "uint16"),
            GgufValueType::Int16 => write!(f, "int16"),
            GgufValueType::Uint32 => write!(f, "uint32"),
            GgufValueType::Int32 => write!(f, "int32"),
            GgufValueType::Float32 => write!(f, "float32"),
            GgufValueType::Bool => write!(f, "bool"),
            GgufValueType::String => write!(f, "string"),
            GgufValueType::Array => write!(f, "array"),
            GgufValueType::Uint64 => write!(f, "uint64"),
            GgufValueType::Int64 => write!(f, "int64"),
            GgufValueType::Float64 => write!(f, "float64"),
        }
    }
}

/// Parsed GGUF metadata scalar value
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
}

impl fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetadataValue::Uint8(v) => write!(f, "{}", v),
            MetadataValue::Int8(v) => write!(f, "{}", v),
            MetadataValue::Uint16(v) => write!(f, "{}", v),
            MetadataValue::Int16(v) => write!(f, "{}", v),
            MetadataValue::Uint32(v) => write!(f, "{}", v),
            MetadataValue::Int32(v) => write!(f, "{}", v),
            MetadataValue::Uint64(v) => write!(f, "{}", v),
            MetadataValue::Int64(v) => write!(f, "{}", v),
            MetadataValue::Float32(v) => write!(f, "{}", v),
            MetadataValue::Float64(v) => write!(f, "{}", v),
            MetadataValue::Bool(v) => write!(f, "{}", v),
            MetadataValue::String(v) => write!(f, "\"{}\"", v),
        }
    }
}

/// A single GGUF metadata key-value entry
#[derive(Debug, Clone)]
pub struct MetadataEntry {
    pub key: String,
    pub value: MetadataValue,
}

/// Collection of parsed GGUF metadata entries
#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub entries: Vec<MetadataEntry>,
}

impl GgufMetadata {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// GGML tensor data type IDs (from llama.cpp ggml.h)
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_1_F16 = 4,
    Q8_0 = 7,
    Q8_1 = 8,
    Q5_0 = 9,
    Q5_1 = 10,
    Q2_K = 11,
    Q3_K = 12,
    Q4_K = 13,
    Q5_K = 14,
    Q6_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31,
    Q4_0_4_8 = 32,
    Q4_0_8_8 = 33,
    TQ1_0 = 34,
    TQ2_0 = 35,
}

impl GgufTensorType {
    pub fn from_u32(id: u32) -> Result<Self, GgufError> {
        match id {
            0 => Ok(GgufTensorType::F32),
            1 => Ok(GgufTensorType::F16),
            2 => Ok(GgufTensorType::Q4_0),
            3 => Ok(GgufTensorType::Q4_1),
            4 => Ok(GgufTensorType::Q4_1_F16),
            7 => Ok(GgufTensorType::Q8_0),
            8 => Ok(GgufTensorType::Q8_1),
            9 => Ok(GgufTensorType::Q5_0),
            10 => Ok(GgufTensorType::Q5_1),
            11 => Ok(GgufTensorType::Q2_K),
            12 => Ok(GgufTensorType::Q3_K),
            13 => Ok(GgufTensorType::Q4_K),
            14 => Ok(GgufTensorType::Q5_K),
            15 => Ok(GgufTensorType::Q6_K),
            16 => Ok(GgufTensorType::IQ2_XXS),
            17 => Ok(GgufTensorType::IQ2_XS),
            18 => Ok(GgufTensorType::IQ3_XXS),
            19 => Ok(GgufTensorType::IQ1_S),
            20 => Ok(GgufTensorType::IQ4_NL),
            21 => Ok(GgufTensorType::IQ3_S),
            22 => Ok(GgufTensorType::IQ2_S),
            23 => Ok(GgufTensorType::IQ4_XS),
            24 => Ok(GgufTensorType::I8),
            25 => Ok(GgufTensorType::I16),
            26 => Ok(GgufTensorType::I32),
            27 => Ok(GgufTensorType::I64),
            28 => Ok(GgufTensorType::F64),
            29 => Ok(GgufTensorType::IQ1_M),
            30 => Ok(GgufTensorType::BF16),
            31 => Ok(GgufTensorType::Q4_0_4_4),
            32 => Ok(GgufTensorType::Q4_0_4_8),
            33 => Ok(GgufTensorType::Q4_0_8_8),
            34 => Ok(GgufTensorType::TQ1_0),
            35 => Ok(GgufTensorType::TQ2_0),
            _ => Err(GgufError::UnsupportedTensorType { type_id: id }),
        }
    }
}

impl fmt::Display for GgufTensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GgufTensorType::F32 => write!(f, "F32"),
            GgufTensorType::F16 => write!(f, "F16"),
            GgufTensorType::Q4_0 => write!(f, "Q4_0"),
            GgufTensorType::Q4_1 => write!(f, "Q4_1"),
            GgufTensorType::Q4_1_F16 => write!(f, "Q4_1_F16"),
            GgufTensorType::Q8_0 => write!(f, "Q8_0"),
            GgufTensorType::Q8_1 => write!(f, "Q8_1"),
            GgufTensorType::Q5_0 => write!(f, "Q5_0"),
            GgufTensorType::Q5_1 => write!(f, "Q5_1"),
            GgufTensorType::Q2_K => write!(f, "Q2_K"),
            GgufTensorType::Q3_K => write!(f, "Q3_K"),
            GgufTensorType::Q4_K => write!(f, "Q4_K"),
            GgufTensorType::Q5_K => write!(f, "Q5_K"),
            GgufTensorType::Q6_K => write!(f, "Q6_K"),
            GgufTensorType::IQ2_XXS => write!(f, "IQ2_XXS"),
            GgufTensorType::IQ2_XS => write!(f, "IQ2_XS"),
            GgufTensorType::IQ3_XXS => write!(f, "IQ3_XXS"),
            GgufTensorType::IQ1_S => write!(f, "IQ1_S"),
            GgufTensorType::IQ4_NL => write!(f, "IQ4_NL"),
            GgufTensorType::IQ3_S => write!(f, "IQ3_S"),
            GgufTensorType::IQ2_S => write!(f, "IQ2_S"),
            GgufTensorType::IQ4_XS => write!(f, "IQ4_XS"),
            GgufTensorType::I8 => write!(f, "I8"),
            GgufTensorType::I16 => write!(f, "I16"),
            GgufTensorType::I32 => write!(f, "I32"),
            GgufTensorType::I64 => write!(f, "I64"),
            GgufTensorType::F64 => write!(f, "F64"),
            GgufTensorType::IQ1_M => write!(f, "IQ1_M"),
            GgufTensorType::BF16 => write!(f, "BF16"),
            GgufTensorType::Q4_0_4_4 => write!(f, "Q4_0_4_4"),
            GgufTensorType::Q4_0_4_8 => write!(f, "Q4_0_4_8"),
            GgufTensorType::Q4_0_8_8 => write!(f, "Q4_0_8_8"),
            GgufTensorType::TQ1_0 => write!(f, "TQ1_0"),
            GgufTensorType::TQ2_0 => write!(f, "TQ2_0"),
        }
    }
}

/// Parsed GGUF tensor descriptor (metadata only, no data loaded)
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub dtype: GgufTensorType,
    pub offset: u64,
}

impl TensorDescriptor {
    /// Total number of elements in the tensor (product of all dimensions).
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// Collection of parsed GGUF tensor descriptors
#[derive(Debug, Clone)]
pub struct GgufTensors {
    pub descriptors: Vec<TensorDescriptor>,
}

impl GgufTensors {
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

/// Error returned when required architecture metadata is missing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingMetadata {
    pub key: String,
}

impl fmt::Display for MissingMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "missing metadata key: {}", self.key)
    }
}

impl std::error::Error for MissingMetadata {}

/// Look up a metadata entry by key, returning None if not found.
fn find_entry<'a>(entries: &'a [MetadataEntry], key: &str) -> Option<&'a MetadataEntry> {
    entries.iter().find(|e| e.key == key)
}

/// Extract a u32 value from metadata. Accepts any integer type source.
fn get_u32(entries: &[MetadataEntry], key: &str) -> Option<u32> {
    let entry = find_entry(entries, key)?;
    match &entry.value {
        MetadataValue::Uint8(v) => Some(*v as u32),
        MetadataValue::Uint16(v) => Some(*v as u32),
        MetadataValue::Uint32(v) => Some(*v),
        MetadataValue::Int32(v) => Some(*v as u32),
        MetadataValue::Uint64(v) => Some(*v as u32),
        MetadataValue::Int64(v) => Some(*v as u32),
        _ => None,
    }
}

/// Extract an i32 value from metadata. Accepts any integer type source.
#[allow(dead_code)]
fn get_i32(entries: &[MetadataEntry], key: &str) -> Option<i32> {
    let entry = find_entry(entries, key)?;
    match &entry.value {
        MetadataValue::Int32(v) => Some(*v),
        MetadataValue::Uint32(v) => Some(*v as i32),
        MetadataValue::Int64(v) => Some(*v as i32),
        MetadataValue::Uint64(v) => Some(*v as i32),
        _ => None,
    }
}

/// Extract a u64 value from metadata. Accepts any integer type source.
fn get_u64(entries: &[MetadataEntry], key: &str) -> Option<u64> {
    let entry = find_entry(entries, key)?;
    match &entry.value {
        MetadataValue::Uint8(v) => Some(*v as u64),
        MetadataValue::Uint16(v) => Some(*v as u64),
        MetadataValue::Uint32(v) => Some(*v as u64),
        MetadataValue::Uint64(v) => Some(*v),
        MetadataValue::Int32(v) => Some(*v as u64),
        MetadataValue::Int64(v) => Some(*v as u64),
        _ => None,
    }
}

/// Extract an f32 value from metadata. Accepts Float32 or Float64 sources.
fn get_f32(entries: &[MetadataEntry], key: &str) -> Option<f32> {
    let entry = find_entry(entries, key)?;
    match &entry.value {
        MetadataValue::Float32(v) => Some(*v),
        MetadataValue::Float64(v) => Some(*v as f32),
        _ => None,
    }
}

/// Extract a String value from metadata.
fn get_string(entries: &[MetadataEntry], key: &str) -> Option<String> {
    let entry = find_entry(entries, key)?;
    match &entry.value {
        MetadataValue::String(v) => Some(v.clone()),
        _ => None,
    }
}

/// Extract a required u32 value, returning MissingMetadata if absent or wrong type.
fn require_u32(entries: &[MetadataEntry], key: &str) -> Result<u32, MissingMetadata> {
    get_u32(entries, key).ok_or_else(|| MissingMetadata {
        key: key.to_string(),
    })
}

/// Extract a required u64 value, returning MissingMetadata if absent or wrong type.
fn require_u64(entries: &[MetadataEntry], key: &str) -> Result<u64, MissingMetadata> {
    get_u64(entries, key).ok_or_else(|| MissingMetadata {
        key: key.to_string(),
    })
}

/// Extract a required f32 value, returning MissingMetadata if absent or wrong type.
#[allow(dead_code)]
fn require_f32(entries: &[MetadataEntry], key: &str) -> Result<f32, MissingMetadata> {
    get_f32(entries, key).ok_or_else(|| MissingMetadata {
        key: key.to_string(),
    })
}

/// Extract a required String value, returning MissingMetadata if absent or wrong type.
fn require_string(entries: &[MetadataEntry], key: &str) -> Result<String, MissingMetadata> {
    get_string(entries, key).ok_or_else(|| MissingMetadata {
        key: key.to_string(),
    })
}

/// Extract an optional String value, returning Ok(None) if absent or wrong type.
#[allow(dead_code)]
fn optional_string(
    entries: &[MetadataEntry],
    key: &str,
) -> Result<Option<String>, MissingMetadata> {
    Ok(get_string(entries, key))
}

/// Extract an optional u32 value, returning Ok(None) if absent or wrong type.
fn optional_u32(entries: &[MetadataEntry], key: &str) -> Result<Option<u32>, MissingMetadata> {
    Ok(get_u32(entries, key))
}

/// Extract an optional u64 value, returning Ok(None) if absent or wrong type.
#[allow(dead_code)]
fn optional_u64(entries: &[MetadataEntry], key: &str) -> Result<Option<u64>, MissingMetadata> {
    Ok(get_u64(entries, key))
}

/// Extract an optional f32 value, returning Ok(None) if absent or wrong type.
fn optional_f32(entries: &[MetadataEntry], key: &str) -> Result<Option<f32>, MissingMetadata> {
    Ok(get_f32(entries, key))
}

/// Extract a required u32 with architecture-specific prefix.
fn arch_u32(entries: &[MetadataEntry], arch: &str, field: &str) -> Result<u32, MissingMetadata> {
    require_u32(entries, &format!("{}.{}", arch, field))
}

/// Extract a required u64 with architecture-specific prefix.
fn arch_u64(entries: &[MetadataEntry], arch: &str, field: &str) -> Result<u64, MissingMetadata> {
    require_u64(entries, &format!("{}.{}", arch, field))
}

/// Extract a required f32 with architecture-specific prefix.
#[allow(dead_code)]
fn arch_f32(entries: &[MetadataEntry], arch: &str, field: &str) -> Result<f32, MissingMetadata> {
    require_f32(entries, &format!("{}.{}", arch, field))
}

/// Extract an optional u32 with architecture-specific prefix.
fn arch_optional_u32(
    entries: &[MetadataEntry],
    arch: &str,
    field: &str,
) -> Result<Option<u32>, MissingMetadata> {
    optional_u32(entries, &format!("{}.{}", arch, field))
}

/// Extract an optional u64 with architecture-specific prefix.
#[allow(dead_code)]
fn arch_optional_u64(
    entries: &[MetadataEntry],
    arch: &str,
    field: &str,
) -> Result<Option<u64>, MissingMetadata> {
    optional_u64(entries, &format!("{}.{}", arch, field))
}

/// Extract an optional f32 with architecture-specific prefix.
fn arch_optional_f32(
    entries: &[MetadataEntry],
    arch: &str,
    field: &str,
) -> Result<Option<f32>, MissingMetadata> {
    optional_f32(entries, &format!("{}.{}", arch, field))
}

/// Extracted architecture metadata for a GGUF model.
#[derive(Debug, Clone)]
pub struct ModelArch {
    /// Value of general.architecture (e.g. "llama", "qwen2")
    pub architecture: String,
    /// Value of general.name if present
    pub name: Option<String>,
    /// Value of tokenizer.ggml.model if present
    pub tokenizer_model: Option<String>,
    /// Number of transformer blocks
    pub block_count: u32,
    /// Context length (maximum sequence length)
    pub context_length: u32,
    /// Hidden / embedding dimension
    pub embedding_length: u32,
    /// Feed-forward (intermediate) dimension
    pub feed_forward_length: u32,
    /// Number of attention heads
    pub attention_head_count: u32,
    /// Number of KV heads (GQA) — None means same as attention_head_count
    pub attention_head_count_kv: Option<u32>,
    /// RoPE frequency base if present
    pub rope_freq_base: Option<f32>,
    /// File type (quantization type) if present
    pub file_type: Option<u32>,
}

impl GgufMetadata {
    /// Extract typed architecture metadata from the parsed GGUF metadata.
    ///
    /// Detects the architecture name from `general.architecture`, then uses
    /// the architecture-specific prefix (e.g. `llama.*`, `qwen2.*`) to look
    /// up the remaining fields.
    ///
    /// Returns `MissingMetadata` if any required field is absent or has an
    /// unexpected type.
    pub fn extract_model_arch(&self) -> Result<ModelArch, MissingMetadata> {
        let entries = &self.entries;
        let architecture = require_string(entries, "general.architecture")?;

        Ok(ModelArch {
            architecture: architecture.clone(),
            name: get_string(entries, "general.name"),
            tokenizer_model: get_string(entries, "tokenizer.ggml.model"),
            block_count: arch_u32(entries, &architecture, "block_count")?,
            context_length: arch_u64(entries, &architecture, "context_length")? as u32,
            embedding_length: arch_u32(entries, &architecture, "embedding_length")?,
            feed_forward_length: arch_u32(entries, &architecture, "feed_forward_length")?,
            attention_head_count: arch_u32(entries, &architecture, "attention.head_count")?,
            attention_head_count_kv: arch_optional_u32(
                entries,
                &architecture,
                "attention.head_count_kv",
            )?,
            rope_freq_base: arch_optional_f32(entries, &architecture, "rope.freq_base")?,
            file_type: get_u32(entries, "general.file_type"),
        })
    }
}

/// Parse a GGUF string from bytes at given offset.
/// Returns (string, bytes_consumed).
/// String format: u64 length prefix + UTF-8 bytes.
fn parse_string(bytes: &[u8], offset: usize) -> Result<(String, usize), GgufError> {
    if offset + 8 > bytes.len() {
        return Err(GgufError::TruncatedMetadata {
            context: "string length",
        });
    }
    let len = u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]) as usize;
    let str_start = offset + 8;
    if str_start + len > bytes.len() {
        return Err(GgufError::TruncatedMetadata {
            context: "string content",
        });
    }
    let s = std::str::from_utf8(&bytes[str_start..str_start + len])
        .map_err(|_| GgufError::InvalidUtf8)?;
    Ok((s.to_string(), 8 + len))
}

/// Parse a GGUF metadata value from bytes at given offset.
/// Returns (value, bytes_consumed).
/// Value format: u32 type ID + typed value.
fn parse_value(bytes: &[u8], offset: usize) -> Result<(MetadataValue, usize), GgufError> {
    if offset + 4 > bytes.len() {
        return Err(GgufError::TruncatedMetadata {
            context: "value type",
        });
    }
    let type_id = u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]);
    let val_offset = offset + 4;

    let value_type = GgufValueType::from_u32(type_id).unwrap_or(GgufValueType::Array);

    match value_type {
        GgufValueType::Uint8 => {
            if val_offset + 1 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "uint8 value",
                });
            }
            Ok((MetadataValue::Uint8(bytes[val_offset]), 5))
        }
        GgufValueType::Int8 => {
            if val_offset + 1 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "int8 value",
                });
            }
            Ok((MetadataValue::Int8(bytes[val_offset] as i8), 5))
        }
        GgufValueType::Uint16 => {
            if val_offset + 2 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "uint16 value",
                });
            }
            let v = u16::from_le_bytes([bytes[val_offset], bytes[val_offset + 1]]);
            Ok((MetadataValue::Uint16(v), 6))
        }
        GgufValueType::Int16 => {
            if val_offset + 2 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "int16 value",
                });
            }
            let v = i16::from_le_bytes([bytes[val_offset], bytes[val_offset + 1]]);
            Ok((MetadataValue::Int16(v), 6))
        }
        GgufValueType::Uint32 => {
            if val_offset + 4 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "uint32 value",
                });
            }
            let v = u32::from_le_bytes([
                bytes[val_offset],
                bytes[val_offset + 1],
                bytes[val_offset + 2],
                bytes[val_offset + 3],
            ]);
            Ok((MetadataValue::Uint32(v), 8))
        }
        GgufValueType::Int32 => {
            if val_offset + 4 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "int32 value",
                });
            }
            let v = i32::from_le_bytes([
                bytes[val_offset],
                bytes[val_offset + 1],
                bytes[val_offset + 2],
                bytes[val_offset + 3],
            ]);
            Ok((MetadataValue::Int32(v), 8))
        }
        GgufValueType::Uint64 => {
            if val_offset + 8 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "uint64 value",
                });
            }
            let v = u64::from_le_bytes([
                bytes[val_offset],
                bytes[val_offset + 1],
                bytes[val_offset + 2],
                bytes[val_offset + 3],
                bytes[val_offset + 4],
                bytes[val_offset + 5],
                bytes[val_offset + 6],
                bytes[val_offset + 7],
            ]);
            Ok((MetadataValue::Uint64(v), 12))
        }
        GgufValueType::Int64 => {
            if val_offset + 8 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "int64 value",
                });
            }
            let v = i64::from_le_bytes([
                bytes[val_offset],
                bytes[val_offset + 1],
                bytes[val_offset + 2],
                bytes[val_offset + 3],
                bytes[val_offset + 4],
                bytes[val_offset + 5],
                bytes[val_offset + 6],
                bytes[val_offset + 7],
            ]);
            Ok((MetadataValue::Int64(v), 12))
        }
        GgufValueType::Float32 => {
            if val_offset + 4 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "float32 value",
                });
            }
            let v = f32::from_le_bytes([
                bytes[val_offset],
                bytes[val_offset + 1],
                bytes[val_offset + 2],
                bytes[val_offset + 3],
            ]);
            Ok((MetadataValue::Float32(v), 8))
        }
        GgufValueType::Float64 => {
            if val_offset + 8 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "float64 value",
                });
            }
            let v = f64::from_le_bytes([
                bytes[val_offset],
                bytes[val_offset + 1],
                bytes[val_offset + 2],
                bytes[val_offset + 3],
                bytes[val_offset + 4],
                bytes[val_offset + 5],
                bytes[val_offset + 6],
                bytes[val_offset + 7],
            ]);
            Ok((MetadataValue::Float64(v), 12))
        }
        GgufValueType::Bool => {
            if val_offset + 1 > bytes.len() {
                return Err(GgufError::TruncatedMetadata {
                    context: "bool value",
                });
            }
            Ok((MetadataValue::Bool(bytes[val_offset] != 0), 5))
        }
        GgufValueType::String => {
            let (s, consumed) = parse_string(bytes, val_offset)?;
            Ok((MetadataValue::String(s), 4 + consumed))
        }
        GgufValueType::Array => Err(GgufError::UnsupportedValueType { type_id }),
    }
}

/// Parse a single GGUF metadata key-value entry.
/// Returns (entry, bytes_consumed).
fn parse_metadata_entry(bytes: &[u8], offset: usize) -> Result<(MetadataEntry, usize), GgufError> {
    let (key, key_consumed) = parse_string(bytes, offset)?;
    let (value, val_consumed) = parse_value(bytes, offset + key_consumed)?;
    Ok((MetadataEntry { key, value }, key_consumed + val_consumed))
}

/// Parse all GGUF metadata entries after the header.
pub fn parse_metadata(bytes: &[u8], kv_count: u64) -> Result<GgufMetadata, GgufError> {
    let mut entries = Vec::with_capacity(kv_count as usize);
    let mut offset = HEADER_SIZE;

    for _ in 0..kv_count {
        let (entry, consumed) = parse_metadata_entry(bytes, offset)?;
        entries.push(entry);
        offset += consumed;
    }

    Ok(GgufMetadata { entries })
}

/// Parse a single GGUF tensor descriptor from bytes at given offset.
/// Returns (descriptor, bytes_consumed).
///
/// GGUF v3 tensor header layout:
///   name: string (u64 length + UTF-8)
///   n_dims: u32
///   shape: [u64; n_dims] (big-to-small order)
///   dtype: u32 (GGML type ID)
///   offset: u64 (absolute file offset)
fn parse_tensor_descriptor(
    bytes: &[u8],
    offset: usize,
) -> Result<(TensorDescriptor, usize), GgufError> {
    let mut pos = offset;

    let (name, consumed) = parse_string(bytes, pos)?;
    pos += consumed;

    if pos + 4 > bytes.len() {
        return Err(GgufError::TruncatedTensor { context: "n_dims" });
    }
    let n_dims = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let shape_size = (n_dims as usize) * 8;
    if pos + shape_size > bytes.len() {
        return Err(GgufError::TruncatedTensor { context: "shape" });
    }
    let mut shape = Vec::with_capacity(n_dims as usize);
    for i in 0..n_dims as usize {
        let val = u64::from_le_bytes(bytes[pos + i * 8..pos + i * 8 + 8].try_into().unwrap());
        shape.push(val);
    }
    pos += shape_size;

    if pos + 4 > bytes.len() {
        return Err(GgufError::TruncatedTensor { context: "dtype" });
    }
    let type_id = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
    pos += 4;

    let dtype = GgufTensorType::from_u32(type_id)?;

    if pos + 8 > bytes.len() {
        return Err(GgufError::TruncatedTensor { context: "offset" });
    }
    let tensor_offset = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
    pos += 8;

    Ok((
        TensorDescriptor {
            name,
            n_dims,
            shape,
            dtype,
            offset: tensor_offset,
        },
        pos - offset,
    ))
}

/// Parse all GGUF tensor descriptors after the metadata region.
pub fn parse_tensors(
    bytes: &[u8],
    tensor_count: u64,
    metadata_end_offset: usize,
) -> Result<GgufTensors, GgufError> {
    let mut descriptors = Vec::with_capacity(tensor_count as usize);
    let mut offset = metadata_end_offset;

    for _ in 0..tensor_count {
        let (desc, consumed) = parse_tensor_descriptor(bytes, offset)?;
        descriptors.push(desc);
        offset += consumed;
    }

    Ok(GgufTensors { descriptors })
}

/// Parse GGUF header + metadata from a byte slice.
pub fn parse_gguf(bytes: &[u8]) -> Result<(GgufHeader, GgufMetadata), GgufError> {
    let header = parse_header(bytes)?;
    let metadata = parse_metadata(bytes, header.metadata_kv_count)?;
    Ok((header, metadata))
}

/// Parse GGUF header + metadata + tensor descriptors from a byte slice.
pub fn parse_gguf_full(bytes: &[u8]) -> Result<(GgufHeader, GgufMetadata, GgufTensors), GgufError> {
    let header = parse_header(bytes)?;
    let metadata = parse_metadata(bytes, header.metadata_kv_count)?;

    let metadata_end_offset = compute_metadata_end_offset(bytes, header.metadata_kv_count)?;
    let tensors = parse_tensors(bytes, header.tensor_count, metadata_end_offset)?;

    Ok((header, metadata, tensors))
}

/// Compute the byte offset immediately after the last metadata entry.
fn compute_metadata_end_offset(bytes: &[u8], kv_count: u64) -> Result<usize, GgufError> {
    let mut offset = HEADER_SIZE;
    for _ in 0..kv_count {
        let (_key, key_consumed) = parse_string(bytes, offset)?;
        offset += key_consumed;

        if offset + 4 > bytes.len() {
            return Err(GgufError::TruncatedMetadata {
                context: "value type",
            });
        }
        let type_id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        offset += 4;
        let val_offset = offset;

        match GgufValueType::from_u32(type_id) {
            Some(GgufValueType::Uint8) | Some(GgufValueType::Int8) | Some(GgufValueType::Bool) => {
                offset += 1;
            }
            Some(GgufValueType::Uint16) | Some(GgufValueType::Int16) => {
                offset += 2;
            }
            Some(GgufValueType::Uint32)
            | Some(GgufValueType::Int32)
            | Some(GgufValueType::Float32) => {
                offset += 4;
            }
            Some(GgufValueType::Uint64)
            | Some(GgufValueType::Int64)
            | Some(GgufValueType::Float64) => {
                offset += 8;
            }
            Some(GgufValueType::String) => {
                let (_s, consumed) = parse_string(bytes, val_offset)?;
                offset += consumed;
            }
            Some(GgufValueType::Array) => {
                let (_array_type_id, consumed) = parse_array_skip(bytes, val_offset)?;
                offset += consumed;
            }
            _ => {
                return Err(GgufError::UnsupportedValueType { type_id });
            }
        }
    }
    Ok(offset)
}

/// Skip over a GGUF array value, returning the element type ID and bytes consumed.
fn parse_array_skip(bytes: &[u8], offset: usize) -> Result<(u32, usize), GgufError> {
    if offset + 4 > bytes.len() {
        return Err(GgufError::TruncatedMetadata {
            context: "array type",
        });
    }
    let type_id = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let mut pos = offset + 4;

    if pos + 8 > bytes.len() {
        return Err(GgufError::TruncatedMetadata {
            context: "array length",
        });
    }
    let len = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
    pos += 8;

    for _ in 0..len {
        match GgufValueType::from_u32(type_id) {
            Some(GgufValueType::Uint8) | Some(GgufValueType::Int8) | Some(GgufValueType::Bool) => {
                pos += 1;
            }
            Some(GgufValueType::Uint16) | Some(GgufValueType::Int16) => {
                pos += 2;
            }
            Some(GgufValueType::Uint32)
            | Some(GgufValueType::Int32)
            | Some(GgufValueType::Float32) => {
                pos += 4;
            }
            Some(GgufValueType::Uint64)
            | Some(GgufValueType::Int64)
            | Some(GgufValueType::Float64) => {
                pos += 8;
            }
            Some(GgufValueType::String) => {
                let (_s, consumed) = parse_string(bytes, pos)?;
                pos += consumed;
            }
            _ => {
                return Err(GgufError::UnsupportedValueType { type_id });
            }
        }
    }

    Ok((type_id, pos - offset))
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
pub fn parse_header_from_path<P: AsRef<std::path::Path>>(path: P) -> Result<GgufHeader, GgufError> {
    let bytes = std::fs::read(path)?;
    parse_header(&bytes)
}

/// Parse GGUF header + metadata from a file on disk.
pub fn parse_gguf_from_path<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<(GgufHeader, GgufMetadata), GgufError> {
    let bytes = std::fs::read(path)?;
    parse_gguf(&bytes)
}

/// Parse GGUF header + metadata + tensor descriptors from a file on disk.
pub fn parse_gguf_full_from_path<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<(GgufHeader, GgufMetadata, GgufTensors), GgufError> {
    let bytes = std::fs::read(path)?;
    parse_gguf_full(&bytes)
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
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

    fn encode_string(s: &str) -> Vec<u8> {
        let len = s.len() as u64;
        let mut buf = Vec::with_capacity(8 + s.len());
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
        buf
    }

    fn make_value(type_id: u32, data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + data.len());
        buf.extend_from_slice(&type_id.to_le_bytes());
        buf.extend_from_slice(data);
        buf
    }

    fn make_entry(key: &str, type_id: u32, data: &[u8]) -> Vec<u8> {
        let mut buf = encode_string(key);
        buf.extend_from_slice(&make_value(type_id, data));
        buf
    }

    fn make_gguf(kv_entries: &[(&str, u32, &[u8])]) -> Vec<u8> {
        let mut buf = make_header(*b"GGUF", 3, 0, kv_entries.len() as u64);
        for (key, type_id, data) in kv_entries {
            buf.extend_from_slice(&make_entry(key, *type_id, data));
        }
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
            Err(GgufError::FileTooShort {
                required,
                available,
            }) => {
                assert_eq!(required, HEADER_SIZE);
                assert_eq!(available, HEADER_SIZE - 1);
            }
            other => panic!("expected FileTooShort, got {:?}", other),
        }
    }

    #[test]
    fn test_file_too_short_empty() {
        match parse_header(&[]) {
            Err(GgufError::FileTooShort {
                required,
                available,
            }) => {
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

    // --- GgufValueType tests ---

    #[test]
    fn test_value_type_roundtrip() {
        for id in 0..=12u32 {
            let vt = GgufValueType::from_u32(id).unwrap();
            assert_eq!(vt.to_u32(), id);
        }
    }

    #[test]
    fn test_value_type_unknown() {
        assert!(GgufValueType::from_u32(13).is_none());
        assert!(GgufValueType::from_u32(100).is_none());
        assert!(GgufValueType::from_u32(u32::MAX).is_none());
    }

    #[test]
    fn test_value_type_display() {
        assert_eq!(format!("{}", GgufValueType::Uint8), "uint8");
        assert_eq!(format!("{}", GgufValueType::String), "string");
        assert_eq!(format!("{}", GgufValueType::Float64), "float64");
        assert_eq!(format!("{}", GgufValueType::Array), "array");
    }

    // --- MetadataValue Display tests ---

    #[test]
    fn test_metadata_value_display_integers() {
        assert_eq!(format!("{}", MetadataValue::Uint8(42)), "42");
        assert_eq!(format!("{}", MetadataValue::Int8(-42)), "-42");
        assert_eq!(format!("{}", MetadataValue::Uint16(65535)), "65535");
        assert_eq!(format!("{}", MetadataValue::Int16(-32768)), "-32768");
        assert_eq!(format!("{}", MetadataValue::Uint32(1_000_000)), "1000000");
        assert_eq!(format!("{}", MetadataValue::Int32(-1_000_000)), "-1000000");
        assert_eq!(
            format!("{}", MetadataValue::Uint64(u64::MAX)),
            format!("{}", u64::MAX)
        );
        assert_eq!(
            format!("{}", MetadataValue::Int64(i64::MIN)),
            format!("{}", i64::MIN)
        );
    }

    #[test]
    fn test_metadata_value_display_floats() {
        assert_eq!(format!("{}", MetadataValue::Float32(1.5)), "1.5");
        assert_eq!(format!("{}", MetadataValue::Float64(3.14)), "3.14");
    }

    #[test]
    fn test_metadata_value_display_bool() {
        assert_eq!(format!("{}", MetadataValue::Bool(true)), "true");
        assert_eq!(format!("{}", MetadataValue::Bool(false)), "false");
    }

    #[test]
    fn test_metadata_value_display_string() {
        assert_eq!(
            format!("{}", MetadataValue::String("hello".to_string())),
            "\"hello\""
        );
        assert_eq!(format!("{}", MetadataValue::String("".to_string())), "\"\"");
    }

    // --- String parsing tests ---

    #[test]
    fn test_parse_string_empty() {
        let bytes = encode_string("");
        let (s, consumed) = parse_string(&bytes, 0).unwrap();
        assert_eq!(s, "");
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_parse_string_hello() {
        let bytes = encode_string("hello");
        let (s, consumed) = parse_string(&bytes, 0).unwrap();
        assert_eq!(s, "hello");
        assert_eq!(consumed, 13);
    }

    #[test]
    fn test_parse_string_unicode() {
        let bytes = encode_string("hello 🌍");
        let (s, consumed) = parse_string(&bytes, 0).unwrap();
        assert_eq!(s, "hello 🌍");
        assert_eq!(consumed, 8 + "hello 🌍".len());
    }

    #[test]
    fn test_parse_string_truncated_length() {
        let bytes = vec![0u8; 4];
        match parse_string(&bytes, 0) {
            Err(GgufError::TruncatedMetadata { context }) => {
                assert_eq!(context, "string length");
            }
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_string_truncated_content() {
        let mut bytes = encode_string("hello world");
        let full_len = bytes.len();
        bytes.truncate(full_len - 3);
        match parse_string(&bytes, 0) {
            Err(GgufError::TruncatedMetadata { context }) => {
                assert_eq!(context, "string content");
            }
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_string_invalid_utf8() {
        let mut buf = Vec::with_capacity(10);
        buf.extend_from_slice(&(2u64).to_le_bytes());
        buf.push(0xFF);
        buf.push(0xFE);
        match parse_string(&buf, 0) {
            Err(GgufError::InvalidUtf8) => {}
            other => panic!("expected InvalidUtf8, got {:?}", other),
        }
    }

    // --- Value parsing tests ---

    #[test]
    fn test_parse_value_uint8() {
        let bytes = make_value(0, &[255]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Uint8(255));
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_parse_value_int8() {
        let bytes = make_value(1, &[0x80]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Int8(-128));
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_parse_value_uint16() {
        let bytes = make_value(2, &[0xFF, 0xFF]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Uint16(65535));
        assert_eq!(consumed, 6);
    }

    #[test]
    fn test_parse_value_int16() {
        let bytes = make_value(3, &[0x00, 0x80]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Int16(-32768));
        assert_eq!(consumed, 6);
    }

    #[test]
    fn test_parse_value_uint32() {
        let bytes = make_value(4, &[0xFF, 0xFF, 0xFF, 0xFF]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Uint32(u32::MAX));
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_parse_value_int32() {
        let bytes = make_value(5, &[0x00, 0x00, 0x00, 0x80]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Int32(i32::MIN));
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_parse_value_uint64() {
        let bytes = make_value(10, &u64::MAX.to_le_bytes());
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Uint64(u64::MAX));
        assert_eq!(consumed, 12);
    }

    #[test]
    fn test_parse_value_int64() {
        let bytes = make_value(11, &i64::MIN.to_le_bytes());
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Int64(i64::MIN));
        assert_eq!(consumed, 12);
    }

    #[test]
    fn test_parse_value_float32() {
        let bytes = make_value(6, &f32::to_le_bytes(3.14159));
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Float32(3.14159));
        assert_eq!(consumed, 8);
    }

    #[test]
    fn test_parse_value_float64() {
        let bytes = make_value(12, &f64::to_le_bytes(2.718281828));
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Float64(2.718281828));
        assert_eq!(consumed, 12);
    }

    #[test]
    fn test_parse_value_bool_true() {
        let bytes = make_value(7, &[1]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Bool(true));
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_parse_value_bool_false() {
        let bytes = make_value(7, &[0]);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Bool(false));
        assert_eq!(consumed, 5);
    }

    #[test]
    fn test_parse_value_bool_nonzero() {
        let bytes = make_value(7, &[42]);
        let (v, _consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::Bool(true));
    }

    #[test]
    fn test_parse_value_string() {
        let str_bytes = encode_string("hello world");
        let mut bytes = make_value(8, &[]);
        bytes.extend_from_slice(&str_bytes);
        let (v, consumed) = parse_value(&bytes, 0).unwrap();
        assert_eq!(v, MetadataValue::String("hello world".to_string()));
        assert_eq!(consumed, 4 + str_bytes.len());
    }

    #[test]
    fn test_parse_value_array_unsupported() {
        let bytes = make_value(9, &[]);
        match parse_value(&bytes, 0) {
            Err(GgufError::UnsupportedValueType { type_id }) => {
                assert_eq!(type_id, 9);
            }
            other => panic!("expected UnsupportedValueType, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_value_unknown_type() {
        let bytes = make_value(99, &[]);
        match parse_value(&bytes, 0) {
            Err(GgufError::UnsupportedValueType { type_id }) => {
                assert_eq!(type_id, 99);
            }
            other => panic!("expected UnsupportedValueType, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_value_truncated_type() {
        let bytes = vec![0u8; 2];
        match parse_value(&bytes, 0) {
            Err(GgufError::TruncatedMetadata { context }) => {
                assert_eq!(context, "value type");
            }
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_value_truncated_data() {
        let bytes = make_value(10, &[0, 0]);
        match parse_value(&bytes, 0) {
            Err(GgufError::TruncatedMetadata { context }) => {
                assert_eq!(context, "uint64 value");
            }
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    // --- Full metadata parsing tests ---

    #[test]
    fn test_parse_metadata_empty() {
        let bytes = make_gguf(&[]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        assert!(metadata.is_empty());
        assert_eq!(metadata.len(), 0);
    }

    #[test]
    fn test_parse_metadata_single_string() {
        let str_bytes = encode_string("llama");
        let bytes = make_gguf(&[("general.architecture", 8, &str_bytes)]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata.entries[0].key, "general.architecture");
        assert_eq!(
            metadata.entries[0].value,
            MetadataValue::String("llama".to_string())
        );
    }

    #[test]
    fn test_parse_metadata_multiple_entries() {
        let name_str = encode_string("MyModel");
        let bytes = make_gguf(&[
            ("general.name", 8, &name_str),
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &42u32.to_le_bytes()),
            ("llama.context_length", 5, &4096u32.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
            ("llama.rope.freq_base", 6, &f32::to_le_bytes(10000.0)),
            ("general.quantization_version", 0, &[2]),
            ("general.file_type", 0, &[0xFF]),
        ]);
        let (header, metadata) = parse_gguf(&bytes).unwrap();
        assert_eq!(header.metadata_kv_count, 9);
        assert_eq!(metadata.len(), 9);
        assert_eq!(metadata.entries[0].key, "general.name");
        assert_eq!(
            metadata.entries[0].value,
            MetadataValue::String("MyModel".to_string())
        );
        assert_eq!(metadata.entries[2].value, MetadataValue::Int32(42));
        assert_eq!(metadata.entries[6].value, MetadataValue::Float32(10000.0));
        assert_eq!(metadata.entries[7].value, MetadataValue::Uint8(2));
        assert_eq!(metadata.entries[8].value, MetadataValue::Uint8(255));
    }

    #[test]
    fn test_parse_metadata_all_scalar_types() {
        let bytes = make_gguf(&[
            ("v.uint8", 0, &[42]),
            ("v.int8", 1, &[200]),
            ("v.uint16", 2, &[0x01, 0x02]),
            ("v.int16", 3, &[0xFF, 0xFF]),
            ("v.uint32", 4, &[0x01, 0x02, 0x03, 0x04]),
            ("v.int32", 5, &[0x00, 0x00, 0x00, 0x80]),
            ("v.float32", 6, &f32::to_le_bytes(1.5)),
            ("v.bool_t", 7, &[1]),
            ("v.string", 8, &encode_string("test")),
            ("v.uint64", 10, &u64::to_le_bytes(12345678901234)),
            ("v.int64", 11, &i64::to_le_bytes(-999)),
            ("v.float64", 12, &f64::to_le_bytes(1.23e10)),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        assert_eq!(metadata.len(), 12);
        assert_eq!(metadata.entries[0].value, MetadataValue::Uint8(42));
        assert_eq!(metadata.entries[1].value, MetadataValue::Int8(-56));
        assert_eq!(metadata.entries[2].value, MetadataValue::Uint16(0x0201));
        assert_eq!(metadata.entries[3].value, MetadataValue::Int16(-1));
        assert_eq!(metadata.entries[4].value, MetadataValue::Uint32(0x04030201));
        assert_eq!(metadata.entries[5].value, MetadataValue::Int32(i32::MIN));
        assert_eq!(metadata.entries[6].value, MetadataValue::Float32(1.5));
        assert_eq!(metadata.entries[7].value, MetadataValue::Bool(true));
        assert_eq!(
            metadata.entries[8].value,
            MetadataValue::String("test".to_string())
        );
        assert_eq!(
            metadata.entries[9].value,
            MetadataValue::Uint64(12345678901234)
        );
        assert_eq!(metadata.entries[10].value, MetadataValue::Int64(-999));
        assert_eq!(metadata.entries[11].value, MetadataValue::Float64(1.23e10));
    }

    #[test]
    fn test_parse_metadata_truncated_key() {
        let mut bytes = make_header(*b"GGUF", 3, 0, 1);
        bytes.extend_from_slice(&(3u64).to_le_bytes());
        bytes.push(b'h');
        match parse_gguf(&bytes) {
            Err(GgufError::TruncatedMetadata { context }) => {
                assert_eq!(context, "string content");
            }
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_metadata_truncated_value() {
        let mut bytes = make_gguf(&[("key", 5, &[])]);
        bytes.truncate(bytes.len() - 1);
        match parse_gguf(&bytes) {
            Err(GgufError::TruncatedMetadata { .. }) => {}
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_metadata_with_array_type() {
        let bytes = make_gguf(&[("arr", 9, &[])]);
        match parse_gguf(&bytes) {
            Err(GgufError::UnsupportedValueType { type_id }) => {
                assert_eq!(type_id, 9);
            }
            other => panic!("expected UnsupportedValueType, got {:?}", other),
        }
    }

    // --- Error display tests for new variants ---

    #[test]
    fn test_error_display_truncated_metadata() {
        let err = GgufError::TruncatedMetadata {
            context: "test context",
        };
        let msg = format!("{}", err);
        assert!(msg.contains("truncated metadata"));
        assert!(msg.contains("test context"));
    }

    #[test]
    fn test_error_display_invalid_utf8() {
        let err = GgufError::InvalidUtf8;
        let msg = format!("{}", err);
        assert!(msg.contains("invalid UTF-8"));
    }

    #[test]
    fn test_error_display_unsupported_value_type() {
        let err = GgufError::UnsupportedValueType { type_id: 99 };
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported"));
        assert!(msg.contains("99"));
    }

    // --- GgufMetadata tests ---

    #[test]
    fn test_metadata_len_and_is_empty() {
        let meta = GgufMetadata { entries: vec![] };
        assert!(meta.is_empty());
        assert_eq!(meta.len(), 0);

        let meta = GgufMetadata {
            entries: vec![MetadataEntry {
                key: "k".to_string(),
                value: MetadataValue::Bool(true),
            }],
        };
        assert!(!meta.is_empty());
        assert_eq!(meta.len(), 1);
    }

    // --- Integration: realistic model metadata ---

    #[test]
    fn test_realistic_llama_metadata() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("general.name", 8, &encode_string("llama-2-7b")),
            ("general.quantization_version", 0, &[2]),
            ("general.file_type", 0, &[0xFF]),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 5, &4096u32.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
            ("llama.attention.head_count_kv", 5, &4u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.rope.freq_base", 6, &f32::to_le_bytes(10000.0)),
            ("llama.rope.dimension_count", 5, &128u32.to_le_bytes()),
            (
                "llama.attention.layer_norm_rms_epsilon",
                6,
                &f32::to_le_bytes(1e-5),
            ),
            ("llama.pooling_type", 5, &0u32.to_le_bytes()),
            ("llama.expert_count", 5, &0u32.to_le_bytes()),
        ]);
        let (header, metadata) = parse_gguf(&bytes).unwrap();
        assert_eq!(header.metadata_kv_count, 15);
        assert_eq!(metadata.len(), 15);

        let arch = &metadata.entries[0];
        assert_eq!(arch.key, "general.architecture");
        assert_eq!(arch.value, MetadataValue::String("llama".to_string()));

        let block_count = &metadata.entries[4];
        assert_eq!(block_count.key, "llama.block_count");
        assert_eq!(block_count.value, MetadataValue::Int32(32));

        let kv_heads = &metadata.entries[8];
        assert_eq!(kv_heads.key, "llama.attention.head_count_kv");
        assert_eq!(kv_heads.value, MetadataValue::Int32(4));

        let rms_eps = &metadata.entries[12];
        assert_eq!(rms_eps.key, "llama.attention.layer_norm_rms_epsilon");
        assert_eq!(rms_eps.value, MetadataValue::Float32(1e-5));
    }

    // --- GgufTensorType tests ---

    #[test]
    fn test_tensor_type_from_u32_known() {
        assert_eq!(GgufTensorType::from_u32(0).unwrap(), GgufTensorType::F32);
        assert_eq!(GgufTensorType::from_u32(1).unwrap(), GgufTensorType::F16);
        assert_eq!(GgufTensorType::from_u32(2).unwrap(), GgufTensorType::Q4_0);
        assert_eq!(GgufTensorType::from_u32(7).unwrap(), GgufTensorType::Q8_0);
        assert_eq!(GgufTensorType::from_u32(9).unwrap(), GgufTensorType::Q5_0);
        assert_eq!(GgufTensorType::from_u32(15).unwrap(), GgufTensorType::Q6_K);
        assert_eq!(GgufTensorType::from_u32(24).unwrap(), GgufTensorType::I8);
        assert_eq!(GgufTensorType::from_u32(30).unwrap(), GgufTensorType::BF16);
    }

    #[test]
    fn test_tensor_type_from_u32_unknown() {
        assert!(matches!(
            GgufTensorType::from_u32(5),
            Err(GgufError::UnsupportedTensorType { type_id: 5 })
        ));
        assert!(matches!(
            GgufTensorType::from_u32(100),
            Err(GgufError::UnsupportedTensorType { type_id: 100 })
        ));
    }

    #[test]
    fn test_tensor_type_display() {
        assert_eq!(format!("{}", GgufTensorType::F32), "F32");
        assert_eq!(format!("{}", GgufTensorType::F16), "F16");
        assert_eq!(format!("{}", GgufTensorType::Q4_0), "Q4_0");
        assert_eq!(format!("{}", GgufTensorType::Q8_0), "Q8_0");
        assert_eq!(format!("{}", GgufTensorType::Q5_K), "Q5_K");
        assert_eq!(format!("{}", GgufTensorType::IQ2_XXS), "IQ2_XXS");
        assert_eq!(format!("{}", GgufTensorType::BF16), "BF16");
        assert_eq!(format!("{}", GgufTensorType::F64), "F64");
    }

    // --- Helper: encode a tensor descriptor to bytes ---

    fn encode_tensor(name: &str, n_dims: u32, shape: &[u64], type_id: u32, offset: u64) -> Vec<u8> {
        let mut buf = encode_string(name);
        buf.extend_from_slice(&n_dims.to_le_bytes());
        for &dim in shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&type_id.to_le_bytes());
        buf.extend_from_slice(&offset.to_le_bytes());
        buf
    }

    // --- Helper: build full GGUF with metadata + tensors ---

    fn make_gguf_full(
        kv_entries: &[(&str, u32, &[u8])],
        tensors: &[(&str, u32, &[u64], u32, u64)],
    ) -> Vec<u8> {
        let mut buf = make_header(*b"GGUF", 3, tensors.len() as u64, kv_entries.len() as u64);
        for (key, type_id, data) in kv_entries {
            buf.extend_from_slice(&make_entry(key, *type_id, data));
        }
        for (name, n_dims, shape, type_id, offset) in tensors {
            buf.extend_from_slice(&encode_tensor(name, *n_dims, shape, *type_id, *offset));
        }
        buf
    }

    // --- Tensor descriptor parsing tests ---

    #[test]
    fn test_parse_tensor_single_dim() {
        let bytes = encode_tensor("test.tensor", 1, &[128], 0, 1000);
        let (desc, consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.name, "test.tensor");
        assert_eq!(desc.n_dims, 1);
        assert_eq!(desc.shape, vec![128]);
        assert_eq!(desc.dtype, GgufTensorType::F32);
        assert_eq!(desc.offset, 1000);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_parse_tensor_two_dims() {
        let bytes = encode_tensor("blk.0.attn_q.weight", 2, &[4096, 4096], 1, 5000);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.name, "blk.0.attn_q.weight");
        assert_eq!(desc.n_dims, 2);
        assert_eq!(desc.shape, vec![4096, 4096]);
        assert_eq!(desc.dtype, GgufTensorType::F16);
        assert_eq!(desc.offset, 5000);
    }

    #[test]
    fn test_parse_tensor_three_dims() {
        let bytes = encode_tensor("complex.tensor", 3, &[64, 128, 256], 2, 10000);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.n_dims, 3);
        assert_eq!(desc.shape, vec![64, 128, 256]);
        assert_eq!(desc.dtype, GgufTensorType::Q4_0);
        assert_eq!(desc.offset, 10000);
    }

    #[test]
    fn test_parse_tensor_four_dims() {
        let bytes = encode_tensor("four.d", 4, &[2, 3, 4, 5], 7, 20000);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.n_dims, 4);
        assert_eq!(desc.shape, vec![2, 3, 4, 5]);
        assert_eq!(desc.dtype, GgufTensorType::Q8_0);
        assert_eq!(desc.offset, 20000);
    }

    #[test]
    fn test_parse_tensor_zero_dims_scalar() {
        let bytes = encode_tensor("scalar", 0, &[], 0, 0);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.name, "scalar");
        assert_eq!(desc.n_dims, 0);
        assert!(desc.shape.is_empty());
        assert_eq!(desc.dtype, GgufTensorType::F32);
        assert_eq!(desc.offset, 0);
    }

    #[test]
    fn test_parse_tensor_empty_name() {
        let bytes = encode_tensor("", 1, &[10], 0, 100);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.name, "");
        assert_eq!(desc.n_dims, 1);
        assert_eq!(desc.shape, vec![10]);
    }

    #[test]
    fn test_parse_tensor_large_offset() {
        let bytes = encode_tensor("big", 1, &[100], 0, u64::MAX);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.offset, u64::MAX);
    }

    #[test]
    fn test_parse_tensor_large_shape() {
        let bytes = encode_tensor("huge", 2, &[u64::MAX, 1], 0, 0);
        let (desc, _consumed) = parse_tensor_descriptor(&bytes, 0).unwrap();
        assert_eq!(desc.shape, vec![u64::MAX, 1]);
    }

    // --- Tensor truncation error tests ---

    #[test]
    fn test_parse_tensor_truncated_name() {
        let mut bytes = encode_string("hello");
        bytes.truncate(8);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        match parse_tensor_descriptor(&bytes, 0) {
            Err(GgufError::TruncatedMetadata { context }) => {
                assert_eq!(context, "string content");
            }
            other => panic!("expected TruncatedMetadata, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_tensor_truncated_n_dims() {
        let mut bytes = encode_string("name");
        bytes.extend_from_slice(&[0x00, 0x00]);
        match parse_tensor_descriptor(&bytes, 0) {
            Err(GgufError::TruncatedTensor { context }) => {
                assert_eq!(context, "n_dims");
            }
            other => panic!("expected TruncatedTensor, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_tensor_truncated_shape() {
        let mut bytes = encode_string("name");
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(100u64).to_le_bytes());
        match parse_tensor_descriptor(&bytes, 0) {
            Err(GgufError::TruncatedTensor { context }) => {
                assert_eq!(context, "shape");
            }
            other => panic!("expected TruncatedTensor, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_tensor_truncated_dtype() {
        let mut bytes = encode_string("name");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(100u64).to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00]);
        match parse_tensor_descriptor(&bytes, 0) {
            Err(GgufError::TruncatedTensor { context }) => {
                assert_eq!(context, "dtype");
            }
            other => panic!("expected TruncatedTensor, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_tensor_truncated_offset() {
        let mut bytes = encode_string("name");
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(100u64).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&[0x00, 0x00, 0x00]);
        match parse_tensor_descriptor(&bytes, 0) {
            Err(GgufError::TruncatedTensor { context }) => {
                assert_eq!(context, "offset");
            }
            other => panic!("expected TruncatedTensor, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_tensor_unsupported_dtype() {
        let bytes = encode_tensor("name", 1, &[10], 999, 0);
        match parse_tensor_descriptor(&bytes, 0) {
            Err(GgufError::UnsupportedTensorType { type_id }) => {
                assert_eq!(type_id, 999);
            }
            other => panic!("expected UnsupportedTensorType, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_tensor_invalid_utf8_name() {
        let mut buf = Vec::with_capacity(10);
        buf.extend_from_slice(&(2u64).to_le_bytes());
        buf.push(0xFF);
        buf.push(0xFE);
        buf.extend_from_slice(&1u32.to_le_bytes());
        match parse_tensor_descriptor(&buf, 0) {
            Err(GgufError::InvalidUtf8) => {}
            other => panic!("expected InvalidUtf8, got {:?}", other),
        }
    }

    // --- parse_tensors tests ---

    #[test]
    fn test_parse_tensors_empty() {
        let bytes = make_gguf_full(&[], &[]);
        let tensors = parse_tensors(&bytes, 0, HEADER_SIZE).unwrap();
        assert!(tensors.is_empty());
        assert_eq!(tensors.len(), 0);
    }

    #[test]
    fn test_parse_tensors_single() {
        let bytes = make_gguf_full(&[], &[("t0", 1, &[64], 0, 100)]);
        let tensors = parse_tensors(&bytes, 1, HEADER_SIZE).unwrap();
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors.descriptors[0].name, "t0");
        assert_eq!(tensors.descriptors[0].shape, vec![64]);
        assert_eq!(tensors.descriptors[0].dtype, GgufTensorType::F32);
        assert_eq!(tensors.descriptors[0].offset, 100);
    }

    #[test]
    fn test_parse_tensors_multiple() {
        let bytes = make_gguf_full(
            &[],
            &[
                ("token_embd.weight", 2, &[4096, 32000], 1, 0),
                ("blk.0.attn_q.weight", 2, &[4096, 4096], 2, 1234),
                ("output_norm.weight", 1, &[4096], 0, 5678),
            ],
        );
        let tensors = parse_tensors(&bytes, 3, HEADER_SIZE).unwrap();
        assert_eq!(tensors.len(), 3);
        assert_eq!(tensors.descriptors[0].name, "token_embd.weight");
        assert_eq!(tensors.descriptors[0].shape, vec![4096, 32000]);
        assert_eq!(tensors.descriptors[1].name, "blk.0.attn_q.weight");
        assert_eq!(tensors.descriptors[1].dtype, GgufTensorType::Q4_0);
        assert_eq!(tensors.descriptors[2].name, "output_norm.weight");
        assert_eq!(tensors.descriptors[2].n_dims, 1);
    }

    // --- TensorDescriptor element_count tests ---

    #[test]
    fn test_element_count_scalar() {
        let desc = TensorDescriptor {
            name: "s".to_string(),
            n_dims: 0,
            shape: vec![],
            dtype: GgufTensorType::F32,
            offset: 0,
        };
        assert_eq!(desc.element_count(), 1);
    }

    #[test]
    fn test_element_count_1d() {
        let desc = TensorDescriptor {
            name: "v".to_string(),
            n_dims: 1,
            shape: vec![128],
            dtype: GgufTensorType::F32,
            offset: 0,
        };
        assert_eq!(desc.element_count(), 128);
    }

    #[test]
    fn test_element_count_2d() {
        let desc = TensorDescriptor {
            name: "m".to_string(),
            n_dims: 2,
            shape: vec![4096, 32000],
            dtype: GgufTensorType::F16,
            offset: 0,
        };
        assert_eq!(desc.element_count(), 4096 * 32000);
    }

    // --- parse_gguf_full integration tests ---

    #[test]
    fn test_parse_gguf_full_empty() {
        let bytes = make_gguf_full(&[], &[]);
        let (header, metadata, tensors) = parse_gguf_full(&bytes).unwrap();
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 0);
        assert_eq!(header.metadata_kv_count, 0);
        assert!(metadata.is_empty());
        assert!(tensors.is_empty());
    }

    #[test]
    fn test_parse_gguf_full_with_metadata_and_tensors() {
        let bytes = make_gguf_full(
            &[
                ("general.architecture", 8, &encode_string("llama")),
                ("general.name", 8, &encode_string("test-model")),
                ("llama.block_count", 5, &2u32.to_le_bytes()),
            ],
            &[
                ("token_embd.weight", 2, &[4096, 32000], 1, 0),
                ("blk.0.attn_q.weight", 2, &[4096, 4096], 0, 10000),
                ("output.weight", 2, &[32000, 4096], 2, 20000),
            ],
        );
        let (header, metadata, tensors) = parse_gguf_full(&bytes).unwrap();
        assert_eq!(header.tensor_count, 3);
        assert_eq!(header.metadata_kv_count, 3);
        assert_eq!(metadata.len(), 3);
        assert_eq!(tensors.len(), 3);
        assert_eq!(
            metadata.entries[0].value,
            MetadataValue::String("llama".to_string())
        );
        assert_eq!(tensors.descriptors[0].name, "token_embd.weight");
        assert_eq!(tensors.descriptors[1].dtype, GgufTensorType::F32);
        assert_eq!(tensors.descriptors[2].offset, 20000);
    }

    #[test]
    fn test_parse_gguf_full_tensors_only() {
        let bytes = make_gguf_full(&[], &[("solo.tensor", 1, &[100], 0, 500)]);
        let (header, metadata, tensors) = parse_gguf_full(&bytes).unwrap();
        assert_eq!(header.metadata_kv_count, 0);
        assert_eq!(header.tensor_count, 1);
        assert!(metadata.is_empty());
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors.descriptors[0].name, "solo.tensor");
    }

    #[test]
    fn test_parse_gguf_full_metadata_only() {
        let bytes = make_gguf_full(&[("key", 8, &encode_string("value"))], &[]);
        let (header, metadata, tensors) = parse_gguf_full(&bytes).unwrap();
        assert_eq!(header.metadata_kv_count, 1);
        assert_eq!(header.tensor_count, 0);
        assert_eq!(metadata.len(), 1);
        assert!(tensors.is_empty());
    }

    // --- Error display tests for tensor error variants ---

    #[test]
    fn test_error_display_truncated_tensor() {
        let err = GgufError::TruncatedTensor { context: "shape" };
        let msg = format!("{}", err);
        assert!(msg.contains("truncated tensor"));
        assert!(msg.contains("shape"));
    }

    #[test]
    fn test_error_display_unsupported_tensor_type() {
        let err = GgufError::UnsupportedTensorType { type_id: 999 };
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported"));
        assert!(msg.contains("999"));
    }

    // --- GgufTensors len/is_empty tests ---

    #[test]
    fn test_tensors_len_and_is_empty() {
        let t = GgufTensors {
            descriptors: vec![],
        };
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);

        let t = GgufTensors {
            descriptors: vec![TensorDescriptor {
                name: "x".to_string(),
                n_dims: 1,
                shape: vec![10],
                dtype: GgufTensorType::F32,
                offset: 0,
            }],
        };
        assert!(!t.is_empty());
        assert_eq!(t.len(), 1);
    }

    // --- Realistic llama model with tensors ---

    #[test]
    fn test_realistic_llama_with_tensors() {
        let bytes = make_gguf_full(
            &[
                ("general.architecture", 8, &encode_string("llama")),
                ("general.name", 8, &encode_string("llama-2-7b")),
                ("llama.block_count", 5, &32u32.to_le_bytes()),
                ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
                ("llama.context_length", 5, &4096u32.to_le_bytes()),
            ],
            &[
                ("token_embd.weight", 2, &[4096, 32000], 1, 0),
                ("blk.0.attn_norm.weight", 1, &[4096], 0, 262144000),
                ("blk.0.attn_q.weight", 2, &[4096, 4096], 1, 262152448),
                ("blk.0.attn_k.weight", 2, &[1024, 4096], 1, 335550592),
                ("blk.0.attn_v.weight", 2, &[1024, 4096], 1, 335572672),
                ("blk.0.attn_output.weight", 2, &[4096, 4096], 1, 335582912),
                ("blk.0.ffn_norm.weight", 1, &[4096], 0, 408981056),
                ("blk.0.ffn_gate.weight", 2, &[11008, 4096], 2, 408985152),
                ("blk.0.ffn_down.weight", 2, &[4096, 11008], 2, 970658304),
                ("blk.0.ffn_up.weight", 2, &[11008, 4096], 2, 970670592),
                ("output_norm.weight", 1, &[4096], 0, 1532356864),
                ("output.weight", 2, &[32000, 4096], 2, 1532360960),
            ],
        );
        let (header, metadata, tensors) = parse_gguf_full(&bytes).unwrap();
        assert_eq!(header.tensor_count, 12);
        assert_eq!(header.metadata_kv_count, 5);
        assert_eq!(metadata.len(), 5);
        assert_eq!(tensors.len(), 12);

        assert_eq!(tensors.descriptors[0].name, "token_embd.weight");
        assert_eq!(tensors.descriptors[0].shape, vec![4096, 32000]);
        assert_eq!(tensors.descriptors[0].dtype, GgufTensorType::F16);

        assert_eq!(tensors.descriptors[3].name, "blk.0.attn_k.weight");
        assert_eq!(tensors.descriptors[3].shape, vec![1024, 4096]);
        assert_eq!(tensors.descriptors[3].dtype, GgufTensorType::F16);

        assert_eq!(tensors.descriptors[7].name, "blk.0.ffn_gate.weight");
        assert_eq!(tensors.descriptors[7].dtype, GgufTensorType::Q4_0);

        assert_eq!(tensors.descriptors[11].name, "output.weight");
        assert_eq!(tensors.descriptors[11].offset, 1532360960);
    }

    // --- Architecture metadata extraction tests ---

    #[test]
    fn test_extract_arch_llama_full() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("general.name", 8, &encode_string("Llama-2-7b")),
            ("tokenizer.ggml.model", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
            ("llama.attention.head_count_kv", 5, &4u32.to_le_bytes()),
            ("llama.rope.freq_base", 6, &f32::to_le_bytes(10000.0)),
            ("general.file_type", 0, &[0xFF]),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let arch = metadata.extract_model_arch().unwrap();

        assert_eq!(arch.architecture, "llama");
        assert_eq!(arch.name, Some("Llama-2-7b".to_string()));
        assert_eq!(arch.tokenizer_model, Some("llama".to_string()));
        assert_eq!(arch.block_count, 32);
        assert_eq!(arch.context_length, 4096);
        assert_eq!(arch.embedding_length, 4096);
        assert_eq!(arch.feed_forward_length, 11008);
        assert_eq!(arch.attention_head_count, 32);
        assert_eq!(arch.attention_head_count_kv, Some(4));
        assert_eq!(arch.rope_freq_base, Some(10000.0));
        assert_eq!(arch.file_type, Some(255));
    }

    #[test]
    fn test_extract_arch_llama_minimal() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let arch = metadata.extract_model_arch().unwrap();

        assert_eq!(arch.architecture, "llama");
        assert_eq!(arch.name, None);
        assert_eq!(arch.tokenizer_model, None);
        assert_eq!(arch.block_count, 32);
        assert_eq!(arch.attention_head_count_kv, None);
        assert_eq!(arch.rope_freq_base, None);
        assert_eq!(arch.file_type, None);
    }

    #[test]
    fn test_extract_arch_qwen2() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("qwen2")),
            ("general.name", 8, &encode_string("Qwen2-7B")),
            ("qwen2.block_count", 5, &32u32.to_le_bytes()),
            ("qwen2.context_length", 11, &32768u64.to_le_bytes()),
            ("qwen2.embedding_length", 5, &4096u32.to_le_bytes()),
            ("qwen2.feed_forward_length", 5, &22016u32.to_le_bytes()),
            ("qwen2.attention.head_count", 5, &32u32.to_le_bytes()),
            ("qwen2.attention.head_count_kv", 5, &2u32.to_le_bytes()),
            ("qwen2.rope.freq_base", 6, &f32::to_le_bytes(1_000_000.0)),
            ("general.file_type", 0, &[2]),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let arch = metadata.extract_model_arch().unwrap();

        assert_eq!(arch.architecture, "qwen2");
        assert_eq!(arch.name, Some("Qwen2-7B".to_string()));
        assert_eq!(arch.block_count, 32);
        assert_eq!(arch.context_length, 32768);
        assert_eq!(arch.embedding_length, 4096);
        assert_eq!(arch.feed_forward_length, 22016);
        assert_eq!(arch.attention_head_count, 32);
        assert_eq!(arch.attention_head_count_kv, Some(2));
        assert_eq!(arch.rope_freq_base, Some(1_000_000.0));
        assert_eq!(arch.file_type, Some(2));
    }

    #[test]
    fn test_extract_arch_missing_architecture() {
        let bytes = make_gguf(&[("llama.block_count", 5, &32u32.to_le_bytes())]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "general.architecture");
    }

    #[test]
    fn test_extract_arch_missing_block_count() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "llama.block_count");
    }

    #[test]
    fn test_extract_arch_missing_context_length() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "llama.context_length");
    }

    #[test]
    fn test_extract_arch_missing_embedding_length() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "llama.embedding_length");
    }

    #[test]
    fn test_extract_arch_missing_ffn_length() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "llama.feed_forward_length");
    }

    #[test]
    fn test_extract_arch_missing_head_count() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "llama.attention.head_count");
    }

    #[test]
    fn test_extract_arch_wrong_type() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 8, &encode_string("not_a_number")),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let err = metadata.extract_model_arch().unwrap_err();
        assert_eq!(err.key, "llama.block_count");
    }

    #[test]
    fn test_extract_arch_int32_block_count() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32i32.to_le_bytes()),
            ("llama.context_length", 11, &4096i64.to_le_bytes()),
            ("llama.embedding_length", 4, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 11, &11008i64.to_le_bytes()),
            ("llama.attention.head_count", 10, &32u64.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let arch = metadata.extract_model_arch().unwrap();

        assert_eq!(arch.block_count, 32);
        assert_eq!(arch.context_length, 4096);
        assert_eq!(arch.embedding_length, 4096);
        assert_eq!(arch.feed_forward_length, 11008);
        assert_eq!(arch.attention_head_count, 32);
    }

    #[test]
    fn test_extract_arch_float64_rope() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("llama")),
            ("llama.block_count", 5, &32u32.to_le_bytes()),
            ("llama.context_length", 11, &4096u64.to_le_bytes()),
            ("llama.embedding_length", 5, &4096u32.to_le_bytes()),
            ("llama.feed_forward_length", 5, &11008u32.to_le_bytes()),
            ("llama.attention.head_count", 5, &32u32.to_le_bytes()),
            ("llama.rope.freq_base", 12, &f64::to_le_bytes(10000.0)),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let arch = metadata.extract_model_arch().unwrap();
        assert_eq!(arch.rope_freq_base, Some(10000.0));
    }

    #[test]
    fn test_extract_arch_unknown_architecture() {
        let bytes = make_gguf(&[
            ("general.architecture", 8, &encode_string("mpt")),
            ("mpt.block_count", 5, &24u32.to_le_bytes()),
            ("mpt.context_length", 11, &8192u64.to_le_bytes()),
            ("mpt.embedding_length", 5, &2048u32.to_le_bytes()),
            ("mpt.feed_forward_length", 5, &8192u32.to_le_bytes()),
            ("mpt.attention.head_count", 5, &16u32.to_le_bytes()),
        ]);
        let (_, metadata) = parse_gguf(&bytes).unwrap();
        let arch = metadata.extract_model_arch().unwrap();

        assert_eq!(arch.architecture, "mpt");
        assert_eq!(arch.block_count, 24);
        assert_eq!(arch.context_length, 8192);
        assert_eq!(arch.embedding_length, 2048);
        assert_eq!(arch.feed_forward_length, 8192);
        assert_eq!(arch.attention_head_count, 16);
        assert_eq!(arch.attention_head_count_kv, None);
    }

    #[test]
    fn test_missing_metadata_display() {
        let err = MissingMetadata {
            key: "test.key".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("missing metadata key"));
        assert!(msg.contains("test.key"));
    }

    #[test]
    fn test_missing_metadata_is_error() {
        let err: Box<dyn std::error::Error> = Box::new(MissingMetadata {
            key: "k".to_string(),
        });
        assert!(format!("{}", err).contains("k"));
    }
}
