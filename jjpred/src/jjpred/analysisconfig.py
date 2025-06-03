from dataclasses import dataclass

import polars as pl
from jjpred.utils.typ import ScalarOrList, normalize_as_list


@dataclass
class RefillConfigInfo:
    """Structure for defining refill configurations at a SKU level."""

    channel: ScalarOrList[str]
    """The channel(s) that this refill configuration applies to."""
    refill_request: int
    """The refill request."""
    sku: ScalarOrList[str]
    """The SKU that this refill configuration applies to."""

    def as_dict(self) -> dict:
        result = {
            k: self.__getattribute__(k)
            if k == "refill_request"
            else normalize_as_list(self.__getattribute__(k))
            for k in ["channel", "refill_request", "sku"]
        }

        return result


@dataclass
class GeneralRefillConfigInfo:
    """Structure for defining refill configurations."""

    channel: ScalarOrList[str]
    """The channel(s) that this refill configuration applies to."""
    refill_request: int
    """The refill request."""
    sku_filter: pl.Expr
    """A polars expression that filters for particular SKUs from the master SKU
    file. For example: ``pl.col.category.eq("BSL")`` or
    ``pl.col.category.eq("HCF0") & pl.col.print.eq("BST")``."""

    def into_refill_config_info(
        self, active_sku_info: pl.DataFrame
    ) -> RefillConfigInfo:
        sku = list(active_sku_info.filter(self.sku_filter)["sku"])
        return RefillConfigInfo(self.channel, self.refill_request, sku)


@dataclass
class AdjustHistorySkuInfo:
    """Sometimes we need to re-use the history of a reference SKU as the history
    of a target SKU."""

    target_sku: str
    ref_sku: str
