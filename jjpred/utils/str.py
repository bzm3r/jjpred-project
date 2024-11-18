"""Utilites for manipulating strings."""


def blank_none(s: str | None, f) -> str:
    if s is None:
        return ""
    else:
        return f(s)


def newln(s: str | None) -> str:
    return blank_none(s, lambda s: f"\n {s}")


def indent(s: str | None, count=2) -> str:
    return blank_none(
        s, lambda s: "\n".join([(" " * count) + x for x in s.split("\n")])
    )


def newln_indent(s: str | None, count=2) -> str:
    return newln(indent(s, count))


def stringify_slice(s: slice) -> str:
    start, stop, step = [
        (repr(val) if val is not None else "")
        for val in [s.start, s.stop, s.step]
    ]
    step_optional = f":{step}" if s.step is not None else ""
    return f"{start}:{stop}{step_optional}"
