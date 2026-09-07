use std::{
    io,
    path::{Component, Path, PathBuf},
};

use crate::cli::PlayPolicy;

/// The maintained human-play selection, not an automatic scan of training output.
pub(crate) const DEFAULT_DEPLOYMENT: DefaultDeployment = DefaultDeployment {
    policy: PlayPolicy::Teacher,
    weights_directory: None,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct DefaultDeployment {
    pub policy: PlayPolicy,
    /// Repository-relative artifact directory for a selected neural policy.
    pub weights_directory: Option<&'static str>,
}

const _: () = {
    match DEFAULT_DEPLOYMENT.policy {
        PlayPolicy::Teacher => assert!(DEFAULT_DEPLOYMENT.weights_directory.is_none()),
        PlayPolicy::Hybrid | PlayPolicy::Neural | PlayPolicy::Tactical => {
            assert!(DEFAULT_DEPLOYMENT.weights_directory.is_some());
        }
    }
    if let Some(directory) = DEFAULT_DEPLOYMENT.weights_directory {
        assert!(!directory.is_empty());
        assert!(directory.len() <= 256);
    }
};

impl DefaultDeployment {
    pub fn resolve(self) -> io::Result<(PlayPolicy, Option<PathBuf>)> {
        match (self.policy, self.weights_directory) {
            (PlayPolicy::Teacher, None) => Ok((self.policy, None)),
            (PlayPolicy::Hybrid | PlayPolicy::Neural | PlayPolicy::Tactical, Some(directory)) => {
                let mut components = Path::new(directory).components();
                if directory.len() > 256
                    || components.next() != Some(Component::Normal("artifacts".as_ref()))
                    || components.clone().next().is_none()
                    || !components.all(|component| matches!(component, Component::Normal(_)))
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "default deployment weights must be a repository-relative directory below artifacts",
                    ));
                }
                Ok((
                    self.policy,
                    Some(Path::new(env!("CARGO_MANIFEST_DIR")).join(directory)),
                ))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "default Teacher must be weights-free; default neural policies must specify weights",
            )),
        }
    }
}
