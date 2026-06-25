def prefer_catalog(
    raw_catalog_ids: list[str],
    catalog: str,
    overlap_map: dict[str, str] | None = None,
) -> list[str]:
    requested = catalog.upper()
    overlap_map = overlap_map or {}
    if requested == "GAIA":
        return [str(value) for value in raw_catalog_ids]

    preferred = []
    for raw_id in raw_catalog_ids:
        value = str(raw_id)
        if value.startswith("GAIA"):
            preferred.append(str(overlap_map.get(value, value)))
        else:
            preferred.append(value)
    return preferred
