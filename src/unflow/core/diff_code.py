import difflib



def get_procedure_changes(source1: str, source2: str) -> dict[int, str]:
    changes = {}
    if source1 != source2:
        diff = difflib.unified_diff(
            source1.splitlines(),
            source2.splitlines(),
            fromfile="previous",
            tofile="current",
            lineterm="",
        )
        # Store the diff as dictionary with line numbers and changes
        changes = dict(enumerate(diff, start=1))
    return changes


def diff_procedure(source1: str, source2: str) -> bool:
    return bool(get_procedure_changes(source1, source2))


