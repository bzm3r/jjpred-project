from dataclasses import dataclass


@dataclass
class RefillConfigInfo:
    """Structure for defining refill configurations at a SKU level."""

    channel: list[str]
    """The channel(s) that this refill configuration applies to."""
    refill_request: int
    """The refill request."""
    sku: str
    """The SKU that this refill configuration applies to."""

    def as_dict(self) -> dict:
        result = {
            k: self.__getattribute__(k)
            for k in ["channel", "refill_request", "sku"]
        }
        # result["channel"] = normalize_as_list(result["channel"])
        return result


@dataclass
class AdjustHistorySkuInfo:
    """Sometimes we need to re-use the history of a reference SKU as the history
    of a target SKU."""

    target_sku: str
    ref_sku: str
