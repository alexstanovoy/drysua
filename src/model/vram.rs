//! A fixed device memory budget, reserved once at startup.
//!
//! candle allocates every tensor from the device's current stream-ordered
//! memory pool. A trainer process replaces that pool with its own: capped at
//! the budget (`maxSize`), filled to the budget before any training tensor
//! exists, and told never to release (release threshold `u64::MAX`). The
//! process footprint is then the budget plus the CUDA context from the first
//! update to the last; an allocation the budget cannot hold fails with
//! CUDA_ERROR_OUT_OF_MEMORY instead of growing the footprint.

use std::sync::{Arc, OnceLock};

use candle_core::cuda_backend::cudarc::driver::{CudaContext, sys};

use super::{ModelError, VramUsage};

/// Pool granularity the budget is rounded up to.
const BUDGET_ALIGNMENT: u64 = 32 << 20;

/// The process's budget pool; at most one exists.
static BUDGET: OnceLock<VramBudget> = OnceLock::new();

/// The reserved pool of one device.
pub(crate) struct VramBudget {
    bytes: u64,
    pool: PoolHandle,
    /// Keeps the primary context, which owns the pool, alive.
    _context: Arc<CudaContext>,
}

/// Pool handle shared across threads; the driver synchronizes pool use.
struct PoolHandle(sys::CUmemoryPool);

// SAFETY: a CUmemoryPool handle is an opaque driver object that every thread
// of the process may use; the driver serializes access to it.
unsafe impl Send for PoolHandle {}
// SAFETY: as above; the handle is never mutated after creation.
unsafe impl Sync for PoolHandle {}

fn check(result: sys::CUresult, what: &str) -> Result<(), ModelError> {
    match result {
        sys::cudaError_enum::CUDA_SUCCESS => Ok(()),
        error => Err(ModelError::Backend(format!("{what}: {error:?}"))),
    }
}

/// Reserves `bytes` (rounded up to the pool granularity) on CUDA device
/// `ordinal` and makes it the device's allocation pool for the rest of the
/// process. Fails, without changing the device, when the GPU cannot hold it.
pub(crate) fn reserve(ordinal: usize, bytes: u64) -> Result<VramUsage, ModelError> {
    if BUDGET.get().is_some() {
        return Err(ModelError::Backend(
            "VRAM budget already reserved".to_owned(),
        ));
    }
    let bytes = bytes.div_ceil(BUDGET_ALIGNMENT).max(1) * BUDGET_ALIGNMENT;
    let context =
        CudaContext::new(ordinal).map_err(|error| ModelError::Backend(error.to_string()))?;
    context
        .bind_to_thread()
        .map_err(|error| ModelError::Backend(error.to_string()))?;
    let device = context.cu_device();
    let size =
        usize::try_from(bytes).map_err(|_| ModelError::Backend("VRAM budget size".into()))?;
    // SAFETY: zeroed props are a valid "nothing requested" value; every field
    // the driver reads is set below.
    let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
    props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
    props.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.__bindgen_anon_1.id =
        i32::try_from(ordinal).map_err(|_| ModelError::Backend("ordinal".into()))?;
    props.maxSize = size;
    let mut pool = std::ptr::null_mut();
    // SAFETY: driver calls on the bound context with valid out-pointers; the
    // probe allocation is freed and synchronized before the pool is installed.
    unsafe {
        check(sys::cuMemPoolCreate(&mut pool, &props), "VRAM budget pool")?;
        let mut keep = u64::MAX;
        check(
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                std::ptr::from_mut(&mut keep).cast(),
            ),
            "VRAM budget release threshold",
        )?;
        // Each lane and the learner allocate on their own stream. Letting one
        // stream take blocks another freed lets small inference blocks split
        // the learner's large ones, and the fragmented pool then needs about
        // three times its live peak; without cross-stream reuse every stream
        // reaches a steady state of its own and the sum of those stays put.
        for attribute in [
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
        ] {
            let mut disabled = 0_i32;
            check(
                sys::cuMemPoolSetAttribute(
                    pool,
                    attribute,
                    std::ptr::from_mut(&mut disabled).cast(),
                ),
                "VRAM budget reuse policy",
            )?;
        }
        let stream = std::ptr::null_mut();
        let mut fill = 0;
        let filled = sys::cuMemAllocFromPoolAsync(&mut fill, size, pool, stream);
        if filled != sys::cudaError_enum::CUDA_SUCCESS {
            sys::cuMemPoolDestroy(pool);
            let (mut free, mut total) = (0, 0);
            sys::cuMemGetInfo_v2(&mut free, &mut total);
            return Err(ModelError::Backend(format!(
                "VRAM budget of {} MiB cannot be reserved: {} MiB of {} MiB free ({filled:?})",
                bytes >> 20,
                free >> 20,
                total >> 20
            )));
        }
        check(
            sys::cuMemFreeAsync(fill, stream),
            "VRAM budget fill release",
        )?;
        check(sys::cuStreamSynchronize(stream), "VRAM budget fill")?;
        // The fill is no tensor: the high-water mark starts at zero.
        let mut zero = 0_u64;
        check(
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                std::ptr::from_mut(&mut zero).cast(),
            ),
            "VRAM budget high-water reset",
        )?;
        check(sys::cuDeviceSetMemPool(device, pool), "VRAM budget install")?;
    }
    let budget = VramBudget {
        bytes,
        pool: PoolHandle(pool),
        _context: context,
    };
    let budget = BUDGET.get_or_init(|| budget);
    budget.usage()
}

/// The installed budget's usage, when a budget was reserved.
pub(crate) fn usage() -> Result<Option<VramUsage>, ModelError> {
    BUDGET.get().map(VramBudget::usage).transpose()
}

impl VramBudget {
    fn usage(&self) -> Result<VramUsage, ModelError> {
        let mut values = [0_u64; 3];
        for (value, attribute) in values.iter_mut().zip([
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        ]) {
            // SAFETY: each attribute is a u64 written into its own element of
            // a live array; the pool lives as long as the process.
            check(
                unsafe {
                    sys::cuMemPoolGetAttribute(
                        self.pool.0,
                        attribute,
                        std::ptr::from_mut(value).cast(),
                    )
                },
                "VRAM budget usage",
            )?;
        }
        let usage = VramUsage {
            budget: self.bytes,
            reserved: values[0],
            used: values[1],
            used_high: values[2],
        };
        // The cap and the release threshold make both hold by construction.
        assert!(usage.reserved <= usage.budget);
        assert!(usage.used_high <= usage.budget);
        Ok(usage)
    }
}

/// The device's current pool: (live bytes, live high-water mark), optionally
/// resetting the mark first. Measurement support for the budget's constants.
#[cfg(test)]
pub(crate) fn current_pool_usage(ordinal: usize, reset_high: bool) -> (u64, u64) {
    let context = CudaContext::new(ordinal).expect("CUDA context");
    context.bind_to_thread().expect("bind");
    let mut pool = std::ptr::null_mut();
    let mut values = [0_u64; 2];
    // SAFETY: driver calls with valid out-pointers on the bound context.
    unsafe {
        check(
            sys::cuDeviceGetMemPool(&mut pool, context.cu_device()),
            "pool",
        )
        .unwrap();
        if reset_high {
            let mut zero = 0_u64;
            check(
                sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                    std::ptr::from_mut(&mut zero).cast(),
                ),
                "reset",
            )
            .unwrap();
        }
        for (value, attribute) in values.iter_mut().zip([
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
        ]) {
            check(
                sys::cuMemPoolGetAttribute(pool, attribute, std::ptr::from_mut(value).cast()),
                "usage",
            )
            .unwrap();
        }
    }
    (values[0], values[1])
}
