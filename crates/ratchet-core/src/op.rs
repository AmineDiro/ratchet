use crate::gpu::{
    BindGroupLayoutDescriptor, ComputePipelineDescriptor, CpuUniform, PipelineLayoutDescriptor,
    PoolError, WgpuDevice, WorkgroupCount,
};
use crate::{ops::*, rvec, CompiledOp, InvariantError, RVec, StorageView, Tensor};
use encase::internal::WriteInto;
use encase::ShaderType;
use std::fmt::Debug;

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum LazyOp {
    Const,
    Matmul(Matmul),
    Binary(Binary),
    Unary(Unary),
    Reindex(Reindex),
    // ---- Everything below this line shouldn't exist ----
    Softmax(Softmax),
    Norm(Norm),
    View(View),             //Should be general class, metadata modification
    Conv(Conv),             //Really it's a matmul
    Select(IndexSelect),    //Can probably be Reindex
    IndexWrite(IndexWrite), //Above 2 should be merged
}

impl LazyOp {
    pub fn key(&self, dst: &Tensor) -> String {
        match self {
            LazyOp::Binary(b) => b.kernel_key(dst),
            LazyOp::Matmul(m) => m.kernel_key(dst),
            LazyOp::Softmax(s) => s.kernel_key(dst),
            LazyOp::Unary(u) => u.kernel_key(dst),
            LazyOp::Reindex(r) => r.kernel_key(dst),
            LazyOp::Norm(n) => n.kernel_key(dst),
            LazyOp::Conv(c) => c.kernel_key(dst),
            LazyOp::Select(s) => s.kernel_key(dst),
            LazyOp::IndexWrite(iw) => iw.kernel_key(dst),
            LazyOp::View(_) => "View".to_string(),
            LazyOp::Const => "Const".to_string(),
        }
    }

    pub fn srcs(&self) -> RVec<&Tensor> {
        match self {
            LazyOp::Binary(b) => b.srcs(),
            LazyOp::Matmul(m) => m.srcs(),
            LazyOp::Softmax(s) => s.srcs(),
            LazyOp::Unary(u) => u.srcs(),
            LazyOp::Reindex(r) => r.srcs(),
            LazyOp::Norm(n) => n.srcs(),
            LazyOp::Conv(c) => c.srcs(),
            LazyOp::Select(s) => s.srcs(),
            LazyOp::IndexWrite(iw) => iw.srcs(),
            LazyOp::View(v) => rvec![v.input()],
            LazyOp::Const => rvec![], //end of the line kid
        }
    }

    pub fn supports_inplace(&self) -> bool {
        match self {
            LazyOp::Binary(b) => b.supports_inplace(),
            LazyOp::Matmul(m) => m.supports_inplace(),
            LazyOp::Softmax(s) => s.supports_inplace(),
            LazyOp::Unary(u) => u.supports_inplace(),
            LazyOp::Reindex(r) => r.supports_inplace(),
            LazyOp::Norm(n) => n.supports_inplace(),
            LazyOp::Conv(c) => c.supports_inplace(),
            LazyOp::Select(s) => s.supports_inplace(),
            LazyOp::IndexWrite(iw) => iw.supports_inplace(),
            LazyOp::View(_v) => true,
            LazyOp::Const => false,
        }
    }

    pub fn is_const(&self) -> bool {
        matches!(self, LazyOp::Const)
    }

    #[track_caller]
    pub fn check_invariants(&self) {
        match self {
            LazyOp::Binary(b) => b.check_invariants(),
            LazyOp::Matmul(m) => m.check_invariants(),
            LazyOp::Softmax(s) => s.check_invariants(),
            LazyOp::Unary(u) => u.check_invariants(),
            LazyOp::Reindex(r) => match r {
                Reindex::Permute(p) => p.check_invariants(),
                Reindex::Slice(s) => s.check_invariants(),
                Reindex::Broadcast(b) => b.check_invariants(),
            },
            LazyOp::Norm(n) => match n {
                Norm::LayerNorm(ln) => ln.check_invariants(),
            },
            LazyOp::Conv(c) => c.check_invariants(),
            LazyOp::Select(s) => s.check_invariants(),
            LazyOp::IndexWrite(iw) => iw.check_invariants(),
            LazyOp::View(v) => v.check_invariants(),
            LazyOp::Const => {}
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum OperationError {
    #[error("Failed to compile operation: {0}")]
    CompileError(String),
    #[error("Failed to get storage layout: {0}")]
    StorageLayoutError(#[from] PoolError),
    #[error(transparent)]
    InvariantError(#[from] InvariantError),
    #[error(transparent)]
    UniformError(#[from] encase::internal::Error),
    #[error(transparent)]
    UnknownError(#[from] anyhow::Error),
}

///A trait for types that are written into uniform buffers, these
///hold the metadata for a shader.
pub trait OpMetadata: Debug + Sized + ShaderType + WriteInto {}

/// # MetaOperation
///
/// Meta Operation is a family of operations that can be compiled into relatively similar shaders.
/// Some types may implement both Operation and MetaOperation, if there is no variance
/// in output shape or invariants between the members of the family.
pub trait MetaOperation: Debug + 'static {
    ///Meta is a struct containing all data written into our uniform buffer.
    ///Typically contains shapes or strides.
    type Meta: OpMetadata;

    /// Return the file stem of the kernel source file.
    fn kernel_key(&self, dst: &Tensor) -> String;

    fn srcs(&self) -> RVec<&Tensor>;

    fn supports_inplace(&self) -> bool {
        false
    }

    /// # Kernel Element
    ///
    /// Determine the largest possible unit data type that can be used (e.g f32, vec2<f32>, vec4<f32>)
    fn kernel_element(&self, dst: &Tensor) -> KernelElement;

    /// # Calculate Dispatch
    ///
    /// Determine required amount of workgroups to execute the operation.
    fn calculate_dispatch(&self, dst: &Tensor) -> Result<WorkgroupCount, OperationError>;

    /// # Storage Bind Group Layout
    ///
    /// Determine the layout of the storage bind group.
    fn storage_bind_group_layout(
        &self,
        inplace: bool,
    ) -> Result<BindGroupLayoutDescriptor, OperationError>;

    fn metadata(
        &self,
        dst: &Tensor,
        kernel_element: &KernelElement,
    ) -> Result<Self::Meta, OperationError>;

    /// # Update
    /// Some operations may require computing additional info once the dst is resolved.
    /// I hate this method.
    fn update(&self, _dst: &Tensor) -> Result<(), OperationError> {
        Ok(())
    }

    fn compile(
        &self,
        dst: &Tensor,
        uniform: &mut CpuUniform,
        device: &WgpuDevice,
        can_inplace: bool,
    ) -> Result<CompiledOp, OperationError> {
        self.update(dst)?;
        let kernel_element = self.kernel_element(dst);
        let meta = self.metadata(dst, &kernel_element)?;
        let offset = uniform.write(&meta)?;

        let workgroup_count = self.calculate_dispatch(dst)?;

        let storage_layout = device
            .get_or_create_bind_group_layout(&self.storage_bind_group_layout(can_inplace)?)?;
        let uniform_layout =
            device.get_or_create_bind_group_layout(&BindGroupLayoutDescriptor::uniform())?;
        let pipeline_layout = device.get_or_create_pipeline_layout(&PipelineLayoutDescriptor {
            entries: rvec![storage_layout, uniform_layout],
        })?;

        let kernel_key = self.kernel_key(dst);
        let pipeline_descriptor = ComputePipelineDescriptor {
            pipeline_layout,
            kernel_key: kernel_key.clone(),
        };
        let pipeline_handle = device.get_or_create_compute_pipeline(&pipeline_descriptor)?;

        //TODO: Not sure i like this call here
        let storage_bind_groups = CompiledOp::create_storage_bind_groups(
            &self.srcs(),
            dst,
            rvec![storage_layout],
            device,
            can_inplace,
            &kernel_key,
        )?;

        Ok(CompiledOp::new(
            pipeline_handle,
            workgroup_count,
            storage_bind_groups,
            offset as _,
            kernel_key,
        ))
    }
}

/// # Operation Guards - Runtime guards for operation correctness.
///
/// Guards should be implemented for all types that will be a node on the high-level CFG.
/// It is used to ensure that the operation is valid and that the resultant tensor is correctly
/// shaped.
///
/// The Rust type system is not sufficient to check all invariants at compile time (we need
/// dependent types). Therefore, we move the checks to runtime.
///
/// All of these methods panic, as they're unrecoverable errors.
pub trait OpGuards {
    #[track_caller]
    fn check_shapes(&self);

    #[track_caller]
    fn check_dtypes(&self);

    // Some operations may have custom invariants to be upheld.
    // e.g reduction dimension being within rank
    #[track_caller]
    fn check_custom(&self) {}
}

/// # Operation
///
/// Operation should be implemented for all types that will be a node on the high-level CFG.
///
/// An Operation is a member of a family of operations, called a MetaOperation, it may be the only
/// member.
pub trait Operation: OpGuards + Debug + 'static {
    /// # Check Invariants
    ///
    /// All operations have some invariants that must be upheld to ensure correctness.
    fn check_invariants(&self) {
        self.check_shapes();
        self.check_dtypes();
        self.check_custom();
    }
    /// # Compute View
    ///
    /// Determine the type, shape & strides of the resultant tensor.
    fn compute_view(&self) -> Result<StorageView, OperationError>;
}
