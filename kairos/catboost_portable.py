class PortableCatBoost:
    @classmethod
    def from_fitted(cls, model):
        result = cls()
        result.__setstate__(model.__getstate__())
        return result

    def __setstate__(self, state):
        self.state = state
        self._model = None

    def __getstate__(self):
        return self.state

    def predict_proba(self, frame):
        if self._model is None:
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            # __setstate__ removes entries; retain the original binary payload.
            model.__setstate__(self.state.copy())
            self._model = model
        return self._model.predict_proba(frame)
