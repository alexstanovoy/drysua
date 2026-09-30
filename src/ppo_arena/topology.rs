//! Cache domains of the host and opt-in thread pinning.
//!
//! A simulation group's lanes and workers exchange every slot each round, so a
//! group is meant to live inside one last-level cache domain (one CCD on a
//! multi-CCD Ryzen). Groups never change results; pinning only places threads.

use crate::PpoError;

/// Bound on parsed domains and CPUs, far above any supported host.
const MAX_CPUS: usize = 1024;

/// CPU sets that share a last-level cache, in order of their lowest CPU.
///
/// Falls back to one domain of every available CPU when the host does not
/// describe its caches.
pub(crate) fn cache_domains() -> Vec<Vec<usize>> {
    let mut domains = Vec::new();
    for cpu in 0..MAX_CPUS {
        let path = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list");
        let Ok(text) = std::fs::read_to_string(path) else {
            if cpu == 0 {
                break;
            }
            continue;
        };
        if let Some(domain) = parse_cpu_list(text.trim())
            && !domains.contains(&domain)
        {
            domains.push(domain);
        }
    }
    if domains.is_empty() {
        let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
        domains.push((0..cpus.min(MAX_CPUS)).collect());
    }
    domains.sort();
    domains
}

/// Parses a sysfs CPU list such as `0-7,16-23`.
fn parse_cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for range in text.split(',') {
        let (first, last): (usize, usize) = match range.split_once('-') {
            Some((first, last)) => (first.parse().ok()?, last.parse().ok()?),
            None => {
                let cpu = range.parse().ok()?;
                (cpu, cpu)
            }
        };
        if first > last || last >= MAX_CPUS {
            return None;
        }
        cpus.extend(first..=last);
    }
    (!cpus.is_empty()).then_some(cpus)
}

/// The CPUs of every simulation group when pinning, one cache domain each.
pub(crate) fn group_cpus(groups: usize, pin: bool) -> Result<Vec<Option<Vec<usize>>>, PpoError> {
    if !pin {
        return Ok(vec![None; groups]);
    }
    let domains = cache_domains();
    if domains.len() < groups {
        return Err(PpoError::InvalidConfig(
            "--pin-threads needs a cache domain per simulation group",
        ));
    }
    Ok(domains.into_iter().take(groups).map(Some).collect())
}

/// Restricts the calling thread to `cpus`.
#[cfg(target_os = "linux")]
pub(crate) fn pin_current_thread(cpus: &[usize]) -> Result<(), PpoError> {
    // SAFETY: cpu_set_t is plain data; CPU_ZERO/CPU_SET only write within it and
    // every index is below MAX_CPUS, which fits the set.
    let result = unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set)
    };
    if result != 0 {
        return Err(PpoError::Model(format!(
            "thread pinning: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

/// Restricts the calling thread to `cpus`.
#[cfg(not(target_os = "linux"))]
pub(crate) fn pin_current_thread(_: &[usize]) -> Result<(), PpoError> {
    Err(PpoError::InvalidConfig(
        "--pin-threads is supported on Linux only",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_parse_ranges_and_reject_malformed_text() {
        assert_eq!(parse_cpu_list("0-3,8"), Some(vec![0, 1, 2, 3, 8]));
        for text in ["", "3-1", "a", "0-2000"] {
            assert_eq!(parse_cpu_list(text), None, "{text}");
        }
        assert!(!cache_domains().is_empty());
    }
}
