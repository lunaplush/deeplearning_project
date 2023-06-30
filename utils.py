def collate_fn(batch):
    return tuple(zip(*batch))


class ProblemClasses():
    class_num = {"nine": 1, "ten": 2, "jack": 3, "queen": 4, "king": 5, "ace": 0}
    class_name = {1: "nine", 2: "ten", 3: "jack", 4: "queen", 5: "king", 0: "ace"}
