"""Arbitration between the prediction engines.

The policy is named, versioned and written down, because the previous behaviour
was none of those things: a hidden rule engine won whenever a made-up confidence
number came out at least as high as the classifier's probability, and the
response still reported the classifier's name and the classifier's accuracy.

Policy ``primary_model_with_rule_fallback_v1``
----------------------------------------------
1. Run the requested statistical engine (``classical``, or ``advanced`` when it
   is installed). This is the primary.
2. Run the symptom-profile rule engine as well, always, so its opinion is on
   the record whether or not it is used.
3. If the primary's probability is at or above ``MODEL_CONFIDENCE_FLOOR``, the
   primary decides.
4. Otherwise, if the rule engine has an opinion, it decides, and the response
   says so.
5. Otherwise the ensemble abstains: the label is ``unknown`` and the knowledge
   base returns its generic "see a clinician" guidance.

The rule engine is a *fallback*, never an override. It speaks only where the
primary has already admitted it does not know, so the two confidence numbers --
which are on different scales and are not comparable -- are never compared.

Whatever happens, ``Decision.decided_by`` names the engine that actually
produced the answer and carries that engine's own measured metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.paths import ENGINE_METRICS_PATH
from app.models.schemas import EngineInfo, EngineMetrics, EngineOpinion
from app.services.prediction_engines import (
    CLASSICAL_ENGINE,
    SEQUENCE_ENGINE,
    SYMPTOM_PROFILE_ENGINE,
    ClassicalModelEngine,
    EngineOutcome,
    PredictionEngine,
    SequenceModelEngine,
    SymptomProfileEngine,
)
from app.utils.text_cleaning import clean_text

ENSEMBLE_POLICY = "primary_model_with_rule_fallback_v1"

# Below this probability the primary model is treated as not knowing.
#
# Not a guess. backend/scripts/train_model.py sweeps candidate floors on the
# validation split, scored by a model fit on the training split only, and writes
# the table into models/classical/metrics.txt. At the time of writing:
#
#   floor  accuracy  macro_f1  rule_used  abstain
#    0.00     0.895     0.869          0        0     <- model only
#    0.20     0.903     0.875          3        0     <- chosen
#    0.30     0.911     0.806         15        2
#    0.40     0.847     0.749         35        2
#
# 0.20 is the only floor that improves on model-only for both accuracy and macro
# F1. 0.30 buys a little more accuracy by handing 15 cases to the rule engine,
# but macro F1 drops 0.07 -- it wins on the common labels and loses the rare
# ones, which is the wrong trade for a triage tool. Re-run the sweep after
# retraining; if no floor beats 0.00 any more, the honest move is to set this to
# 0.0 and let the rule engine be an advisory opinion only.
MODEL_CONFIDENCE_FLOOR = 0.20

UNKNOWN_LABEL = "unknown"

STATISTICAL_ENGINE_KEYS = (CLASSICAL_ENGINE, SEQUENCE_ENGINE)


@dataclass(frozen=True)
class Decision:
    predicted_disease: str
    confidence: float
    decided_by: EngineInfo
    policy: str
    policy_reason: str
    abstained: bool
    opinions: list[EngineOpinion]


class DiseasePredictionService:
    def __init__(self) -> None:
        self.engines: dict[str, PredictionEngine] = {
            CLASSICAL_ENGINE: ClassicalModelEngine(),
            SEQUENCE_ENGINE: SequenceModelEngine(),
            SYMPTOM_PROFILE_ENGINE: SymptomProfileEngine(),
        }

    # -- public API --------------------------------------------------------

    def predict(self, text: str, model_key: str = CLASSICAL_ENGINE) -> Decision:
        cleaned = clean_text(text)
        primary_key = self.resolve_primary(model_key)

        primary = self.engines[primary_key]
        rule = self.engines[SYMPTOM_PROFILE_ENGINE]

        primary_outcome = primary.predict(cleaned)
        rule_outcome = rule.predict(cleaned)
        opinions = [self._to_opinion(primary_outcome), self._to_opinion(rule_outcome)]

        primary_confidence = primary_outcome.confidence or 0.0

        if primary_outcome.disease is not None and primary_confidence >= MODEL_CONFIDENCE_FLOOR:
            return Decision(
                predicted_disease=primary_outcome.disease,
                confidence=primary_confidence,
                decided_by=self.engine_info(primary_key),
                policy=ENSEMBLE_POLICY,
                policy_reason=(
                    f"The {primary_key} engine reported {primary_confidence:.3f}, at or above the "
                    f"{MODEL_CONFIDENCE_FLOOR} confidence floor, so it decided."
                ),
                abstained=False,
                opinions=opinions,
            )

        if rule_outcome.disease is not None:
            return Decision(
                predicted_disease=rule_outcome.disease,
                confidence=rule_outcome.confidence or 0.0,
                decided_by=self.engine_info(SYMPTOM_PROFILE_ENGINE),
                policy=ENSEMBLE_POLICY,
                policy_reason=(
                    f"The {primary_key} engine reported {primary_confidence:.3f}, below the "
                    f"{MODEL_CONFIDENCE_FLOOR} confidence floor, so the symptom-profile rule "
                    f"engine answered instead. The confidence shown is that engine's match "
                    f"share, not a probability."
                ),
                abstained=False,
                opinions=opinions,
            )

        return Decision(
            predicted_disease=UNKNOWN_LABEL,
            confidence=primary_confidence,
            decided_by=self.engine_info(primary_key),
            policy=ENSEMBLE_POLICY,
            policy_reason=(
                f"The {primary_key} engine reported {primary_confidence:.3f}, below the "
                f"{MODEL_CONFIDENCE_FLOOR} confidence floor, and no symptom profile matched. "
                f"The ensemble abstained rather than guess."
            ),
            abstained=True,
            opinions=opinions,
        )

    def resolve_primary(self, model_key: str) -> str:
        """Which statistical engine actually runs, given what was requested."""
        if model_key in STATISTICAL_ENGINE_KEYS and self.engines[model_key].is_available():
            return model_key
        return CLASSICAL_ENGINE

    def list_engines(self) -> list[EngineInfo]:
        return [self.engine_info(key) for key in self.engines]

    def engine_info(self, key: str) -> EngineInfo:
        engine = self.engines[key]
        measured = engine.metrics()
        return EngineInfo(
            key=engine.key,
            name=engine.name,
            family=engine.family,
            description=engine.description,
            confidence_meaning=engine.confidence_meaning,
            is_available=engine.is_available(),
            is_default=engine.key == CLASSICAL_ENGINE,
            accuracy=measured.get("accuracy"),
            macro_f1=measured.get("macro_f1"),
            metrics=EngineMetrics(
                accuracy=measured.get("accuracy"),
                accuracy_ci_95=measured.get("accuracy_ci_95"),
                macro_f1=measured.get("macro_f1"),
                external_use_case_accuracy=measured.get("external_use_case_accuracy"),
                cross_source_transfer_recall=measured.get("cross_source_transfer_recall"),
                basis=measured.get(
                    "basis",
                    "not evaluated; run backend/scripts/evaluate_engines.py",
                ),
                evaluated_at=measured.get("evaluated_at"),
                source_file=str(ENGINE_METRICS_PATH.name),
            ),
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_opinion(outcome: EngineOutcome) -> EngineOpinion:
        return EngineOpinion(
            engine=outcome.engine_key,
            disease=outcome.disease,
            confidence=outcome.confidence,
            available=outcome.available,
            detail=outcome.detail,
            matched_symptoms=list(outcome.matched_symptoms),
        )
