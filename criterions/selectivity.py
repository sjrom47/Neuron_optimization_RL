from criterions.criterion import Criterion


class SelectivityCriterion(Criterion):
    @property
    def requires_multiple_responses(self):
        return True

    def evaluate(self, responses):
        # Placeholder for selectivity calculation
        # TODO:  implement actual selectivity formula
        selectivity = ...
        return selectivity
