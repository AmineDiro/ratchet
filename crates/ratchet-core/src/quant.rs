use num::integer::div_floor;
use num_traits::{AsPrimitive, Float, FromPrimitive, Zero};

use std::fmt::Debug;

use crate::{
    dtype::Quantized, gpu::STORAGE_BUFFER_ALIGN, DType, Device, Tensor, Q4_KF, Q4_KH, Q8_0F, Q8_0H,
};

/// Quantizer
///
/// Packs weights into our custom quantization formats.
#[derive(Debug, derive_new::new)]
pub struct Quantizer {
    format: Quantization,
}

#[inline]
fn storage_align<T>(n: usize) -> usize {
    let size_t = core::mem::size_of::<T>();
    let nbytes = n * size_t;
    let aligned = if nbytes % STORAGE_BUFFER_ALIGN != 0 {
        nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
    } else {
        nbytes
    };
    aligned / size_t
}

pub fn quantize_inner<Q: Quantized>(matrix: &[Q::FP], elements: usize) -> Vec<u32> {
    println!("quantize_inner");
    assert_eq!(elements % Q::PACK_SIZE, 0);
    assert_eq!(elements % Q::GROUP_SIZE, 0);

    let qmatrix_len = elements / Q::PACK_SIZE;
    let amatrix_len = elements / Q::GROUP_SIZE;

    let mut quantized_matrix = vec![0u32; storage_align::<u32>(qmatrix_len)];
    let mut absmax_matrix = vec![Q::FP::zero(); storage_align::<Q::FP>(amatrix_len)];
    let mut block_absmax = Q::FP::neg_infinity();

    for i in (0..elements).step_by(Q::PACK_SIZE) {
        if i % Q::GROUP_SIZE == 0 {
            let amax = matrix[i..i + Q::GROUP_SIZE]
                .iter()
                .fold(Q::FP::neg_infinity(), |acc, &x| acc.max(x.abs()));
            let d = amax / Q::FP::from_i32((1 << 7) - 1).unwrap();
            block_absmax = d;
        }
        for j in 0..Q::PACK_SIZE {
            let packed_value: i32 =
                ((matrix[i + j] / block_absmax).round().as_() & Q::MASK) << (j * Q::LSHIFT);
            quantized_matrix[i / Q::PACK_SIZE] |= packed_value as u32;
        }
        absmax_matrix[i / Q::GROUP_SIZE] = block_absmax;
    }
    quantized_matrix.append(&mut unsafe { std::mem::transmute(absmax_matrix) });

    quantized_matrix
}

pub fn quantize<Q: Quantized>(tensor: &Tensor) -> Tensor {
    match (tensor.dt(), Q::dt()) {
        (DType::F32, DType::Q8_0F(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q8_0F(Q8_0F::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (DType::F32, DType::Q4_KF(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q4_KF(Q4_KF::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (DType::F16, DType::Q8_0H(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q8_0H(Q8_0H::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (DType::F16, DType::Q4_KH(_)) => {
            let matrix = tensor.to_vec::<Q::FP>().unwrap();
            unsafe {
                Tensor::from_quantized(
                    quantize_inner::<Q>(&matrix, tensor.shape().numel()),
                    DType::Q4_KH(Q4_KH::default()),
                    tensor.shape().clone(),
                    Device::CPU,
                )
            }
        }
        (dt, qdt) => panic!("Unsupported dtype combination {dt}, {qdt}"),
    }
}

fn dequantize_inner<Q: Quantized>(quantized: &[u8], numel: usize) -> Vec<Q::FP> {
    println!("dequantize_inner");
    let num_q = numel / Q::PACK_SIZE;
    let num_q_bytes = num_q * std::mem::size_of::<u32>();
    let aligner = |numel: usize, size_t: usize| -> usize {
        let nbytes = numel * size_t;

        if nbytes % STORAGE_BUFFER_ALIGN != 0 {
            nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
        } else {
            nbytes
        }
    };
    let aligned_q_bytes = aligner(num_q, std::mem::size_of::<u32>());

    let num_absmax = numel / Q::GROUP_SIZE;
    let num_absmax_bytes = num_absmax * std::mem::size_of::<Q::FP>();

    let quantized_matrix = bytemuck::cast_slice::<u8, u32>(&quantized[..num_q_bytes]);
    let absmax_matrix = bytemuck::cast_slice::<u8, Q::FP>(
        &quantized[aligned_q_bytes..aligned_q_bytes + num_absmax_bytes],
    );

    let mut dequantized = vec![Q::FP::zero(); numel];
    for i in (0..numel).step_by(Q::PACK_SIZE) {
        let block_absmax = absmax_matrix[div_floor(i, Q::GROUP_SIZE)];
        let packed_value = quantized_matrix[div_floor(i, Q::PACK_SIZE)] as i32;
        for j in 0..Q::PACK_SIZE {
            dequantized[i + j] = Q::FP::from_i32(
                (packed_value << (Q::LSHIFT * (Q::PACK_SIZE - j - 1))) >> Q::RSHIFT,
            )
            .unwrap()
                * block_absmax;
        }
    }

    dequantized
}

pub fn dequantize(quantized: Tensor) -> Tensor {
    return match quantized.dt() {
        DType::Q8_0F(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q8_0F>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        DType::Q4_KF(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q4_KF>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        DType::Q8_0H(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q8_0H>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        DType::Q4_KH(_) => {
            let elements = quantized.shape().numel();
            let original_shape = quantized.shape().clone();
            let raw_bytes = unsafe { quantized.into_bytes().unwrap() };
            let dequantized = dequantize_inner::<Q4_KH>(&raw_bytes, elements);
            Tensor::from_data(&dequantized, original_shape, Device::CPU)
        }
        dt => panic!("Unsupported dtype {dt}"),
    };
}

impl Quantizer {
    /// Quantizes a float 32 tensor into a packed uint32 tensor.
    pub fn sint8_quantize(&self, tensor: Tensor) -> Tensor {
        let numel = tensor.shape().numel();
        let pack_size = self.format.pack_size();
        let group_size = self.format.group_size();

        assert!(numel % pack_size == 0 && numel % group_size == 0);
        assert!(tensor.dt() == DType::F32); //TODO: f16, bf16
                                            //TODO: check if tensor is contiguous
        let qmatrix_len = numel / pack_size;
        let amatrix_len = numel / group_size;

        //returns the aligned number of ELEMENTS
        let aligner = |numel: usize, size_t: usize| -> usize {
            let nbytes = numel * size_t;
            let aligned = if nbytes % STORAGE_BUFFER_ALIGN != 0 {
                nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
            } else {
                nbytes
            };
            aligned / size_t
        };

        let mut quantized_matrix = vec![0u32; aligner(qmatrix_len, std::mem::size_of::<u32>())];
        let mut absmax_matrix = vec![0f32; aligner(amatrix_len, std::mem::size_of::<f32>())];

        let mut block_absmax = f32::NEG_INFINITY;

        let matrix = tensor.to_vec::<f32>().unwrap();

        for i in (0..numel).step_by(pack_size) {
            if i % group_size == 0 {
                let amax = matrix[i..i + group_size]
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x.abs()));
                let d = amax / ((1 << 7) - 1) as f32;
                block_absmax = d;
            }
            let packed_value: i32 = ((matrix[i] / block_absmax).round() as i32 & 0xFF)
                | (((matrix[i + 1] / block_absmax).round() as i32 & 0xFF) << 8)
                | (((matrix[i + 2] / block_absmax).round() as i32 & 0xFF) << 16)
                | (((matrix[i + 3] / block_absmax).round() as i32 & 0xFF) << 24);
            quantized_matrix[i / pack_size] = packed_value as u32;
            absmax_matrix[i / group_size] = block_absmax;
        }
        quantized_matrix.append(&mut unsafe { std::mem::transmute(absmax_matrix) });
        unsafe {
            Tensor::from_quantized(
                quantized_matrix,
                DType::Q8_0F(Q8_0F::default()),
                tensor.shape().clone(),
                Device::CPU,
            )
        }
    }

    pub fn sint8_dequantize(&self, quantized: Tensor) -> Tensor {
        assert!(matches!(quantized.dt(), DType::Q8_0F(_)));
        let numel = quantized.shape().numel();
        let original_shape = quantized.shape().clone();
        let aligner = |numel: usize, size_t: usize| -> usize {
            let nbytes = numel * size_t;

            if nbytes % STORAGE_BUFFER_ALIGN != 0 {
                nbytes + STORAGE_BUFFER_ALIGN - nbytes % STORAGE_BUFFER_ALIGN
            } else {
                nbytes
            }
        };

        let pack_size = self.format.pack_size();
        let group_size = self.format.group_size();

        let num_q = numel / pack_size;
        let num_q_bytes = num_q * std::mem::size_of::<u32>();
        let aligned_q_bytes = aligner(num_q, std::mem::size_of::<u32>());

        let num_absmax = numel / group_size;
        let num_absmax_bytes = num_absmax * std::mem::size_of::<f32>();

        let raw_bytes = unsafe { quantized.into_bytes().unwrap() };

        let quantized_matrix = bytemuck::cast_slice::<u8, u32>(&raw_bytes[..num_q_bytes]);
        let absmax_matrix = bytemuck::cast_slice::<u8, f32>(
            &raw_bytes[aligned_q_bytes..aligned_q_bytes + num_absmax_bytes],
        );

        let mut dequantized = vec![0.0f32; numel];

        for i in (0..numel).step_by(pack_size) {
            let block_absmax = absmax_matrix[div_floor(i, group_size)];
            let packed_value = quantized_matrix[div_floor(i, pack_size)] as i32;
            dequantized[i] = ((packed_value << 24) >> 24) as f32 * block_absmax;
            dequantized[i + 1] = ((packed_value << 16) >> 24) as f32 * block_absmax;
            dequantized[i + 2] = ((packed_value << 8) >> 24) as f32 * block_absmax;
            dequantized[i + 3] = (packed_value >> 24) as f32 * block_absmax;
        }

        Tensor::from_data(dequantized, original_shape, Device::CPU)
    }

    pub fn sint4_quantize<F: Float + AsPrimitive<i32> + Debug>(
        matrix: &[F],
        K: usize,
        N: usize,
    ) -> (Vec<u32>, F) {
        assert!(matrix.len() == K * N);
        assert!(matrix.len() % 4 == 0);
        let pack_size = 8;
        let mut quantized_matrix = vec![0u32; K * N / pack_size];

        let absmax = matrix.iter().fold(F::zero(), |acc, &x| acc.max(x.abs()));
        let sf = F::from(7.).unwrap();

        for i in (0..(K * N)).step_by(pack_size) {
            let packed_value: i32 = ((matrix[i] / absmax * sf).round().as_() & 0xF)
                | (((matrix[i + 1] / absmax * sf).round().as_() & 0xF) << 4)
                | (((matrix[i + 2] / absmax * sf).round().as_() & 0xF) << 8)
                | (((matrix[i + 3] / absmax * sf).round().as_() & 0xF) << 12)
                | (((matrix[i + 4] / absmax * sf).round().as_() & 0xF) << 16)
                | (((matrix[i + 5] / absmax * sf).round().as_() & 0xF) << 20)
                | (((matrix[i + 6] / absmax * sf).round().as_() & 0xF) << 24)
                | (((matrix[i + 7] / absmax * sf).round().as_() & 0xF) << 28);
            quantized_matrix[i / pack_size] = packed_value as u32
        }
        (quantized_matrix, absmax)
    }

    pub fn sint4_dequantize(quantized_matrix: &[u32], absmax: f32, K: usize, N: usize) -> Vec<f32> {
        let pack_size = 8;
        let mut matrix = vec![0.0; K * N];

        for i in (0..(K * N)).step_by(pack_size) {
            let packed_value = quantized_matrix[div_floor(i, pack_size)] as i32;
            matrix[i] = ((packed_value << 28) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 1] = ((packed_value << 24) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 2] = ((packed_value << 20) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 3] = ((packed_value << 16) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 4] = ((packed_value << 12) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 5] = ((packed_value << 8) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 6] = ((packed_value << 4) >> 28) as f32 / 7.0 * absmax;
            matrix[i + 7] = (packed_value >> 28) as f32 / 7.0 * absmax;
        }

        matrix
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Quantization {
    None,
    SInt8,
    SInt4,
}

impl Quantization {
    pub fn pack_size(&self) -> usize {
        match self {
            Quantization::None => 1,
            Quantization::SInt8 => 4,
            Quantization::SInt4 => 8,
        }
    }

    pub fn group_size(&self) -> usize {
        match self {
            Quantization::None => 1,
            Quantization::SInt8 => 32,
            Quantization::SInt4 => 8,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        dequantize, quantize, quantize_inner, shape, Device, Quantization, Quantizer, Tensor,
        Q4_KF, Q8_0F,
    };

    #[test]
    pub fn test_sint8_qdq() {
        let ground = Tensor::randn::<f32>(shape![64, 64], Device::CPU);

        // Old api
        let quantizer = Quantizer::new(Quantization::SInt8);
        let q1 = quantizer.sint8_quantize(ground.deep_clone());
        let dq1 = quantizer.sint8_dequantize(q1.deep_clone());

        // New api
        let q2 = quantize::<Q8_0F>(&ground);
        let dq2 = dequantize(q2.deep_clone());

        let q1_raw = unsafe { q1.deep_clone().into_bytes().unwrap() };
        let q2_raw = unsafe { q2.deep_clone().into_bytes().unwrap() };
        assert_eq!(q1_raw, q2_raw);
        if q1_raw == q2_raw {
            println!("SInt8 quantization is correct");
        }

        dq1.all_close(&dq2, 1e-3, 1e-3).unwrap();
    }

    #[test]
    pub fn test_sint4_qdq() {
        let ground = Tensor::randn::<f32>(shape![64, 64], Device::CPU);

        // Old api
        let data = ground.to_vec::<f32>().unwrap();
        let (q1, absmax) = Quantizer::sint4_quantize::<f32>(&data, 64, 64);
        let dq1 = Quantizer::sint4_dequantize(&q1, absmax, 64, 64);

        // New api
        let q2 = quantize_inner::<Q4_KF>(&data, 64 * 64);
        //let dq2 = dequantize(q2.deep_clone());

        for (a, b) in q1.iter().zip(q2.iter()) {
            if a != b {
                println!("{} {}", a, b);
            }
        }
        /*
        let dq2_vec = dq2.to_vec::<f32>().unwrap();
        for (a, b) in dq1.iter().zip(dq2_vec.iter()) {
            if (a - b).abs() >= 1e-3 {
                println!("{} {}", a, b);
            }
        }
         */
    }
}
