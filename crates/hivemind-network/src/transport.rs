use hivemind_core::{NodeId, PipelineId, Result, HivemindError, Tensor};

/// Streams a tensor activation to the next node in the pipeline over QUIC.
///
/// TODO: implement with `libp2p-quic` + a custom framing protocol.
/// Wire format:
///   - 4 bytes: pipeline_id length
///   - N bytes: pipeline_id (UUID as string)
///   - 4 bytes: shape dimensions count
///   - 4 * D bytes: shape dimensions (u32 each)
///   - 4 * numel bytes: f32 data (little-endian)
pub async fn forward_activations(
    _destination: NodeId,
    _pipeline_id: PipelineId,
    _tensor: Tensor,
) -> Result<Tensor> {
    // TODO: implement
    // 1. Open a QUIC stream to `destination`
    // 2. Serialize and stream the tensor
    // 3. Await the response activation tensor
    // 4. Deserialize and return
    Err(HivemindError::Network(
        "activation forwarding not yet implemented".into(),
    ))
}

/// Serializes a tensor into the Hivemind wire format.
///
/// Layout: [ndim: u32][dim0: u32]...[dimN: u32][data: f32 * numel]
pub fn serialize_tensor(tensor: &Tensor) -> Vec<u8> {
    let ndim = tensor.shape.len() as u32;
    let numel = tensor.numel();
    let mut buf = Vec::with_capacity(4 + 4 * tensor.shape.len() + 4 * numel);

    buf.extend_from_slice(&ndim.to_le_bytes());
    for &d in &tensor.shape {
        buf.extend_from_slice(&(d as u32).to_le_bytes());
    }
    for &v in &tensor.data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Deserializes a tensor from the Hivemind wire format.
pub fn deserialize_tensor(buf: &[u8]) -> Result<Tensor> {
    if buf.len() < 4 {
        return Err(HivemindError::Network("buffer too short for tensor header".into()));
    }
    let ndim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let header_len = 4 + 4 * ndim;
    if buf.len() < header_len {
        return Err(HivemindError::Network("buffer too short for tensor shape".into()));
    }

    let mut shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let start = 4 + i * 4;
        let dim = u32::from_le_bytes(buf[start..start + 4].try_into().unwrap()) as usize;
        shape.push(dim);
    }

    let numel: usize = shape.iter().product();
    let data_len = header_len + 4 * numel;
    if buf.len() < data_len {
        return Err(HivemindError::Network("buffer too short for tensor data".into()));
    }

    let mut data = Vec::with_capacity(numel);
    for i in 0..numel {
        let start = header_len + i * 4;
        let v = f32::from_le_bytes(buf[start..start + 4].try_into().unwrap());
        data.push(v);
    }

    Ok(Tensor::new(data, shape))
}
