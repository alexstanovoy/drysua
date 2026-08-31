#![allow(
    clippy::float_arithmetic,
    reason = "profiling reports rates and averages as floating-point diagnostics"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use bota_proto::{HeroId, MapId, Pick, SlotId, Team, TickMode};
use bota_server::game::{MatchConfig, World};
use clap::Parser;
use drysua::{Arena, ArenaConfig, ArenaError, PolicyDevice, PpoSmokeConfig, run_ppo_smoke_on};

static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

struct CountingAllocator;

// SAFETY: Every operation delegates to `System` with the unchanged pointer and layout.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: The caller supplies the layout required by `GlobalAlloc`.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: The caller returns the pointer with its original layout.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // SAFETY: The caller supplies the original layout and requested replacement size.
        let replacement = unsafe { System.realloc(pointer, layout, size) };
        if !replacement.is_null() {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(size as u64, Ordering::Relaxed);
        }
        replacement
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Parser)]
#[command(about = "Bounded release profiling before large-scale training")]
struct Arguments {
    #[arg(long, default_value_t = 2)]
    arenas: usize,
    #[arg(long, default_value_t = 500)]
    ticks: usize,
    #[arg(long, default_value_t = 0)]
    training_updates: u32,
    #[arg(long, default_value_t = 8)]
    rollout_decisions: usize,
    #[arg(long, default_value_t = false)]
    probe_nvfp4: bool,
    #[arg(long, default_value_t = false)]
    cuda_training: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    if !(1..=16).contains(&arguments.arenas)
        || !(1..=20_000).contains(&arguments.ticks)
        || arguments.training_updates > 2
    {
        return Err("profile bounds are arenas=1..16, ticks=1..20000, updates=0..2".into());
    }
    if !(1..=64).contains(&arguments.rollout_decisions) {
        return Err("profile rollout decisions are outside 1..=64".into());
    }
    profile_simulation(arguments.arenas, arguments.ticks)?;
    profile_world(arguments.arenas, arguments.ticks);
    profile_projection(arguments.arenas, arguments.ticks);
    profile_training(
        "cpu_training",
        arguments.training_updates,
        arguments.arenas.min(16),
        arguments.rollout_decisions,
        PolicyDevice::Cpu,
    )?;
    if arguments.cuda_training {
        profile_cuda_training(
            arguments.training_updates,
            arguments.arenas.min(16),
            arguments.rollout_decisions,
        )?;
    }
    if arguments.probe_nvfp4 {
        print_nvfp4_probe();
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn print_nvfp4_probe() {
    match drysua::probe_nvfp4(0) {
        Ok(()) => println!("profile precision=nvfp4 status=available"),
        Err(error) => println!("profile precision=nvfp4 status=unavailable error={error}"),
    }
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn print_nvfp4_probe() {
    println!("profile precision=nvfp4 status=unavailable error=CUDA feature is disabled");
}

fn profile_world(worlds: usize, ticks: usize) {
    let mut worlds = (0..worlds)
        .map(|seed| profile_world_for(seed as u64))
        .collect::<Vec<_>>();
    reset_allocations();
    let started = Instant::now();
    for _ in 0..ticks {
        for world in &mut worlds {
            black_box(world.advance(&[]));
        }
    }
    print_profile("world", (worlds.len() * ticks) as u64, started.elapsed());
}

fn profile_projection(worlds: usize, ticks: usize) {
    let mut worlds = (0..worlds)
        .map(|seed| profile_world_for(seed as u64))
        .collect::<Vec<_>>();
    reset_allocations();
    let started = Instant::now();
    for _ in 0..ticks {
        for world in &mut worlds {
            black_box(world.advance(&[]));
            black_box(world.view(Team::Radiant));
            black_box(world.view(Team::Dire));
        }
    }
    print_profile(
        "world_and_projection",
        (worlds.len() * ticks) as u64,
        started.elapsed(),
    );
}

fn profile_simulation(arenas: usize, ticks: usize) -> Result<(), Box<dyn std::error::Error>> {
    let mut worlds = (0..arenas)
        .map(|index| new_arena(index as u64))
        .collect::<Result<Vec<_>, _>>()?;
    reset_allocations();
    let started = Instant::now();
    let mut advanced = 0u64;
    for _ in 0..ticks {
        for (index, arena) in worlds.iter_mut().enumerate() {
            match arena.step(&[None, None]) {
                Ok(step) => {
                    black_box(step);
                    advanced += 1;
                }
                Err(ArenaError::MatchOver) => {
                    *arena = new_arena(index as u64 + advanced)?;
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    let elapsed = started.elapsed();
    print_profile("simulation", advanced, elapsed);
    Ok(())
}

fn profile_training(
    name: &str,
    updates: u32,
    environments: usize,
    rollout_decisions: usize,
    device: PolicyDevice,
) -> Result<(), Box<dyn std::error::Error>> {
    if updates == 0 {
        return Ok(());
    }
    reset_allocations();
    let started = Instant::now();
    let report = run_ppo_smoke_on(
        PpoSmokeConfig {
            updates,
            environments,
            rollout_decisions,
            epochs: 1,
            minibatch: environments * rollout_decisions,
            seed: 19_001,
            map: MapId(1),
        },
        device,
    )?;
    print_profile(name, report.transitions as u64, started.elapsed());
    Ok(())
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn profile_cuda_training(
    updates: u32,
    environments: usize,
    rollout_decisions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    profile_training(
        "cuda_training",
        updates,
        environments,
        rollout_decisions,
        PolicyDevice::Cuda { ordinal: 0 },
    )
}

#[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
fn profile_cuda_training(_: u32, _: usize, _: usize) -> Result<(), Box<dyn std::error::Error>> {
    Err("CUDA profiling requires the cuda feature on Linux or Windows".into())
}

fn new_arena(seed: u64) -> Result<Arena, ArenaError> {
    Arena::new(ArenaConfig {
        seats: 2,
        map: MapId(1),
        seed: 19_000 + seed,
    })
    .map(|(arena, _)| arena)
}

fn profile_world_for(seed: u64) -> World {
    let mut master_key = [0; 32];
    master_key[..8].copy_from_slice(&seed.to_le_bytes());
    let config = MatchConfig {
        match_id: seed,
        master_key,
        picks: vec![
            Pick {
                slot: SlotId(0),
                team: Team::Radiant,
                hero: HeroId(2),
            },
            Pick {
                slot: SlotId(1),
                team: Team::Dire,
                hero: HeroId(2),
            },
        ],
        map: MapId(1),
        tick_rate: 30,
        mode: TickMode::Lockstep,
        ack_timeout_ticks: 0,
    };
    let mut world = World::for_match(&config, config.rng());
    world.advance(&[]);
    world
}

fn reset_allocations() {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
}

fn print_profile(name: &str, units: u64, elapsed: std::time::Duration) {
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);
    let bytes = ALLOCATED_BYTES.load(Ordering::Relaxed);
    let seconds = elapsed.as_secs_f64();
    println!(
        "profile phase={name} units={units} elapsed_ms={:.3} units_per_second={:.0} allocations={} allocations_per_unit={:.2} bytes={} bytes_per_unit={:.0}",
        seconds * 1_000.0,
        units as f64 / seconds,
        allocations,
        allocations as f64 / units as f64,
        bytes,
        bytes as f64 / units as f64,
    );
}
