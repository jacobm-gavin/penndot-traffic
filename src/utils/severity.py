def derive_serious_from_max_severity(max_severity_value):
    """
    Map MAX_SEVERITY_LEVEL to SERIOUS indicator.
    From dictionary:
      0 – Property Damage Only
      1 – Fatal
      2 – Suspected Serious Injury
      3 – Suspected Minor Injury
      4 – Possible Injury
      8 – Injury – Unknown Severity
      9 – Unknown if Injured

    Returns 1 if value in {1, 2}, else 0. Handles strings and ints; returns 0 for unknowns.
    """
    if max_severity_value is None:
        return 0
    try:
        v = int(str(max_severity_value).strip())
    except Exception:
        return 0
    return 1 if v in (1, 2) else 0

