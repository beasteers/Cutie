from collections import Counter


class ObjectInfo:
    """
    Store meta information for an object
    """
    def __init__(self, id: int):
        self.id = id
        self.poke_count = 0  # count number of detections missed
        self.label_count = Counter()
        self.seen_count = 0
        self.exist_count = 0

    def update_label(self, label):
        self.label_count.update([label])

    @property
    def label(self):
        if not len(self.label_count):
            return None
        return self.label_count.most_common(1)[0][0]

    @property
    def scores(self):
        if not len(self.label_count):
            return None
        scores = self.label_count.most_common()
        total = sum(s for l, s in scores)
        return {l: s/total for l, s in scores}

    def poke(self) -> None:
        self.poke_count += 1
        self.exist_count += 1

    def unpoke(self) -> None:
        self.poke_count = 0
        self.seen_count += 1
        self.exist_count += 1

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if type(other) == int:
            return self.id == other
        return self.id == other.id

    def __repr__(self):
        return f'(ID: {self.id})'
