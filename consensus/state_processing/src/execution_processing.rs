use super::signature_sets::{execution_envelope_signature_set, get_pubkey_from_state};
use crate::per_block_processing::compute_timestamp_at_slot;
use crate::per_block_processing::errors::{BlockProcessingError, ExecutionEnvelopeError};
use crate::VerifySignatures;
use tree_hash::TreeHash;
use types::{BeaconState, ChainSpec, EthSpec, Hash256, SignedExecutionEnvelope};

pub fn process_execution_envelope<E: EthSpec>(
    state: &mut BeaconState<E>,
    signed_envelope: SignedExecutionEnvelope<E>,
    spec: &ChainSpec,
    verify_signatures: VerifySignatures,
) -> Result<(), BlockProcessingError> {
    if verify_signatures.is_true() {
        block_verify!(
            execution_envelope_signature_set(
                state,
                |i| get_pubkey_from_state(state, i),
                &signed_envelope,
                spec
            )?
            .verify(),
            ExecutionEnvelopeError::BadSignature.into()
        )
    }

    let envelope = signed_envelope.message();
    let payload = &envelope.payload;
    let previous_state_root = state.canonical_root()?;
    if state.latest_block_header().state_root == Hash256::default() {
        *state.latest_block_header_mut().state_root = *previous_state_root;
    }

    // Verify consistency with the beacon block
    block_verify!(
        envelope.tree_hash_root() == state.latest_block_header().tree_hash_root(),
        ExecutionEnvelopeError::LatestBlockHeaderMismatch {
            envelope_root: envelope.tree_hash_root(),
            block_header_root: state.latest_block_header().tree_hash_root(),
        }
        .into()
    );

    // Verify consistency with the committed bid
    let committed_bid = state.latest_execution_bid()?;
    block_verify!(
        envelope.builder_index == committed_bid.builder_index,
        ExecutionEnvelopeError::BuilderIndexMismatch {
            committed_bid: committed_bid.builder_index,
            envelope: envelope.builder_index,
        }
        .into()
    );
    block_verify!(
        committed_bid.blob_kzg_commitments_root == envelope.blob_kzg_commitments.tree_hash_root(),
        ExecutionEnvelopeError::BlobKzgCommitmentsRootMismatch {
            committed_bid: committed_bid.blob_kzg_commitments_root,
            envelope: envelope.blob_kzg_commitments.tree_hash_root(),
        }
        .into()
    );

    if !envelope.payment_withheld {
        // Verify the withdrawals root
        block_verify!(
            payload.withdrawals.tree_hash_root() == state.latest_withdrawals_root()?,
            ExecutionEnvelopeError::WithdrawalsRootMismatch {
                state: state.latest_withdrawals_root()?,
                envelope: payload.withdrawals.tree_hash_root(),
            }
            .into()
        );

        // Verify the gas limit
        block_verify!(
            payload.gas_limit == committed_bid.gas_limit,
            ExecutionEnvelopeError::GasLimitMismatch {
                committed_bid: committed_bid.gas_limit,
                envelope: payload.gas_limit,
            }
            .into()
        );

        block_verify!(
            committed_bid.block_hash == payload.block_hash,
            ExecutionEnvelopeError::BlockHashMismatch {
                committed_bid: committed_bid.block_hash,
                envelope: payload.block_hash,
            }
            .into()
        );

        // Verify consistency of the parent hash with respect to the previous execution payload
        block_verify!(
            payload.parent_hash == state.latest_block_hash()?,
            ExecutionEnvelopeError::ParentHashMismatch {
                state: state.latest_block_hash()?,
                envelope: payload.parent_hash,
            }
            .into()
        );

        // Verify prev_randao
        block_verify!(
            payload.prev_randao == *state.get_randao_mix(state.current_epoch())?,
            ExecutionEnvelopeError::PrevRandaoMismatch {
                state: *state.get_randao_mix(state.current_epoch())?,
                envelope: payload.prev_randao,
            }
            .into()
        );

        // Verify the timestamp
        let state_timestamp = compute_timestamp_at_slot(state, state.slot(), spec)?;
        block_verify!(
            payload.timestamp == state_timestamp,
            ExecutionEnvelopeError::TimestampMismatch {
                state: state_timestamp,
                envelope: payload.timestamp,
            }
            .into()
        );

        // Verify the commitments are under limit
        block_verify!(
            envelope.blob_kzg_commitments.len() <= E::max_blob_commitments_per_block(),
            ExecutionEnvelopeError::BlobLimitExceeded {
                max: E::max_blob_commitments_per_block(),
                envelope: envelope.blob_kzg_commitments.len(),
            }
            .into()
        );
    }

    Ok(())
}
