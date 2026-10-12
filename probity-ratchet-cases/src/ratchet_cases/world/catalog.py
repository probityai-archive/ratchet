"""The product catalog: products, their variants and prices. Read-only."""

from __future__ import annotations

import copy

from ratchet_cases.world.system import JSON, System, ToolDef, params, string


class Catalog(System):
    name = "catalog"

    @staticmethod
    def default_state() -> JSON:
        return {"products": {}}

    def tools(self) -> list[ToolDef]:
        return [
            ToolDef(
                "catalog_get_product",
                "Read a product: its description and every variant (sku, options, price, availability).",
                params({"product_id": string("Product id, for example prod_tee.")}),
                self._get,
            ),
        ]

    def _get(self, args: JSON, actor: str) -> JSON:
        return {"product": copy.deepcopy(self.row("products", args["product_id"], "product"))}

    def variant(self, sku: str) -> tuple[str, JSON] | None:
        for product_id, product in self.state["products"].items():
            found = product["variants"].get(sku)
            if found is not None:
                return product_id, found
        return None
