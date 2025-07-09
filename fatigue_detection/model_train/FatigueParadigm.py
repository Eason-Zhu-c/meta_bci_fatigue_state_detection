from metabci.brainda.paradigms.base import BaseParadigm

class Fatigue(BaseParadigm):
    def is_valid(self, dataset):
        ret = True
        if dataset.paradigm != "fatigue":
            ret = False
        return ret
