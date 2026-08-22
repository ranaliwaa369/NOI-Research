"""Auditable corrective updating for NOI associative memory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from src.memory.records import AssociativeMemoryRecord
from src.memory.temporal_memory import TemporalAssociativeMemory


class CorrectiveUpdateError(ValueError):
    """Raised when a corrective-memory update is invalid."""


def _vector_sha256(vector: tuple[float, ...]) -> str:
    """Return a stable SHA-256 fingerprint for a numeric vector."""

    serialized = json.dumps(
        list(vector),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(serialized).hexdigest()


@dataclass(frozen=True, slots=True)
class CorrectionAuditRecord:
    """Immutable evidence describing one corrective update."""

    correction_id: str
    memory_id: str
    previous_odor_item_id: str
    corrected_odor_item_id: str
    previous_context_hash: str
    corrected_context_hash: str
    corrected_at_utc: datetime
    reason: str
    protocol_hash: str
    resulting_correction_count: int

    def __post_init__(self) -> None:
        if not self.correction_id.strip():
            raise ValueError("correction_id must not be empty.")

        if not self.memory_id.strip():
            raise ValueError("memory_id must not be empty.")

        if not self.reason.strip():
            raise ValueError("A correction reason is required.")

        if (
            self.corrected_at_utc.tzinfo is None
            or self.corrected_at_utc.utcoffset() is None
        ):
            raise ValueError(
                "corrected_at_utc must be timezone-aware."
            )

        hash_fields = {
            "previous_context_hash": self.previous_context_hash,
            "corrected_context_hash": self.corrected_context_hash,
            "protocol_hash": self.protocol_hash,
        }

        for name, value in hash_fields.items():
            if len(value) != 64:
                raise ValueError(
                    f"{name} must be a 64-character SHA-256 value."
                )

        if self.resulting_correction_count < 1:
            raise ValueError(
                "resulting_correction_count must be at least 1."
            )


class CorrectiveMemoryUpdater:
    """Apply non-silent corrections and retain an immutable audit log."""

    def __init__(
        self,
        *,
        memory: TemporalAssociativeMemory,
        protocol_hash: str,
    ) -> None:
        if len(protocol_hash) != 64:
            raise CorrectiveUpdateError(
                "protocol_hash must be a 64-character SHA-256 value."
            )

        self._memory = memory
        self._protocol_hash = protocol_hash
        self._audit_log: list[CorrectionAuditRecord] = []
        self._used_correction_ids: set[str] = set()

    @property
    def audit_log(self) -> tuple[CorrectionAuditRecord, ...]:
        """Return an immutable view of completed correction events."""

        return tuple(self._audit_log)

    def apply(
        self,
        *,
        correction_id: str,
        memory_id: str,
        corrected_at_utc: datetime,
        reason: str,
        corrected_odor_item_id: str | None = None,
        corrected_context_vector: Iterable[float] | None = None,
        corrected_strength: float | None = None,
    ) -> CorrectionAuditRecord:
        """Correct one association without erasing its prior identity."""

        if not correction_id.strip():
            raise CorrectiveUpdateError(
                "correction_id must not be empty."
            )

        if correction_id in self._used_correction_ids:
            raise CorrectiveUpdateError(
                f"Duplicate correction_id: {correction_id}"
            )

        if not reason.strip():
            raise CorrectiveUpdateError(
                "A correction reason is required."
            )

        if (
            corrected_at_utc.tzinfo is None
            or corrected_at_utc.utcoffset() is None
        ):
            raise CorrectiveUpdateError(
                "corrected_at_utc must be timezone-aware."
            )

        previous = self._memory.get(memory_id)

        if corrected_at_utc < previous.updated_at_utc:
            raise CorrectiveUpdateError(
                "A correction cannot precede the previous update."
            )

        new_odor_item_id = (
            corrected_odor_item_id
            if corrected_odor_item_id is not None
            else previous.odor_item_id
        )

        if not new_odor_item_id.strip():
            raise CorrectiveUpdateError(
                "corrected_odor_item_id must not be empty."
            )

        if corrected_context_vector is None:
            new_context_vector = previous.context_vector
        else:
            new_context_vector = tuple(
                float(value)
                for value in corrected_context_vector
            )

        new_strength = (
            previous.strength
            if corrected_strength is None
            else corrected_strength
        )

        replacement = AssociativeMemoryRecord(
            memory_id=previous.memory_id,
            context_vector=new_context_vector,
            odor_item_id=new_odor_item_id,
            created_at_utc=previous.created_at_utc,
            updated_at_utc=corrected_at_utc,
            strength=new_strength,
            correction_count=previous.correction_count + 1,
            active=previous.active,
        )

        audit_record = CorrectionAuditRecord(
            correction_id=correction_id,
            memory_id=previous.memory_id,
            previous_odor_item_id=previous.odor_item_id,
            corrected_odor_item_id=replacement.odor_item_id,
            previous_context_hash=_vector_sha256(
                previous.context_vector
            ),
            corrected_context_hash=_vector_sha256(
                replacement.context_vector
            ),
            corrected_at_utc=corrected_at_utc,
            reason=reason,
            protocol_hash=self._protocol_hash,
            resulting_correction_count=replacement.correction_count,
        )

        self._memory.replace(replacement)
        self._used_correction_ids.add(correction_id)
        self._audit_log.append(audit_record)

        return audit_record