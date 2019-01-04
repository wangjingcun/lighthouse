use crate::{BeaconBlock, Hash256};

pub trait BeaconBlockReader {
    fn slot(&self) -> u64;
    fn parent_root(&self) -> Hash256;
    fn state_root(&self) -> Hash256;
    fn canonical_root(&self) -> Hash256;
    fn into_beacon_block(self) -> Option<BeaconBlock>;
}

impl BeaconBlockReader for BeaconBlock {
    fn slot(&self) -> u64 {
        self.slot
    }

    fn parent_root(&self) -> Hash256 {
        self.parent_root
    }

    fn state_root(&self) -> Hash256 {
        self.state_root
    }

    fn canonical_root(&self) -> Hash256 {
        self.canonical_root()
    }

    fn into_beacon_block(self) -> Option<BeaconBlock> {
        Some(self)
    }
}
