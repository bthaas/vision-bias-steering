def load_evaluation_task(task):
    if task == "winogenerated":
        from .winogenerated import Winogenerated
        return Winogenerated()
    else:
        raise Exception(f"Requested dataset does not exist: {task}")
