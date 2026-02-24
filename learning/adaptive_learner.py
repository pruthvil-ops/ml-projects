import pandas as pd


class AdaptiveLearner:
    def __init__(self) -> None:
        self.history: list[dict] = []

    def record_feedback(self, prediction: str, actual: str, confidence: float) -> None:
        self.history.append(
            {
                "prediction": prediction,
                "actual": actual,
                "confidence": float(confidence),
                "is_correct": prediction == actual,
            }
        )

    def summary(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame(columns=["prediction", "actual", "confidence", "is_correct"])
        return pd.DataFrame(self.history)


adaptive_learner = AdaptiveLearner()
