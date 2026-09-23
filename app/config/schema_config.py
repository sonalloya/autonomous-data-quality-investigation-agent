"""
app/config/schema_config.py
----------------------------
Schema configuration management module.

Provides loading and access utilities for expected dataset schema definitions.
Supports sales schema by default and can load schemas for other datasets
(e.g., customers, products) dynamically from JSON files or Python dictionaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default directory for schema JSON definitions
SCHEMAS_DIR = Path(__file__).parent / "schemas"


@dataclass
class ColumnSchema:
    """Expected schema rule for a single column."""
    name: str
    logical_type: str  # 'string', 'integer', 'float', 'datetime', 'boolean'
    required: bool = True
    nullable: bool = True
    expected_format: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_value_exclusive: Optional[float] = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ColumnSchema:
        return cls(
            name=data["name"],
            logical_type=data.get("logical_type", "string").lower(),
            required=data.get("required", True),
            nullable=data.get("nullable", True),
            expected_format=data.get("expected_format"),
            allowed_values=data.get("allowed_values"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            min_value_exclusive=data.get("min_value_exclusive"),
            description=data.get("description", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "logical_type": self.logical_type,
            "required": self.required,
            "nullable": self.nullable,
            "expected_format": self.expected_format,
            "allowed_values": self.allowed_values,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_value_exclusive": self.min_value_exclusive,
            "description": self.description,
        }


@dataclass
class SchemaDefinition:
    """Container for full dataset schema definition."""
    dataset_name: str
    version: str = "1.0"
    description: str = ""
    columns: List[ColumnSchema] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SchemaDefinition:
        cols = [ColumnSchema.from_dict(c) for c in data.get("columns", [])]
        return cls(
            dataset_name=data.get("dataset_name", "unknown"),
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            columns=cols,
        )

    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> SchemaDefinition:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Schema file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "description": self.description,
            "columns": [col.to_dict() for col in self.columns],
        }

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        for col in self.columns:
            if col.name == name:
                return col
        return None


def get_sales_schema() -> SchemaDefinition:
    """Load the default sales dataset schema definition."""
    sales_json_path = SCHEMAS_DIR / "sales_schema.json"
    if sales_json_path.exists():
        return SchemaDefinition.from_file(sales_json_path)
    
    # Fallback in-memory definition if json file is missing
    return SchemaDefinition(
        dataset_name="sales",
        version="1.0",
        description="Expected schema definition for sales transactions dataset",
        columns=[
            ColumnSchema(name="transaction_id", logical_type="string", required=True, nullable=False),
            ColumnSchema(name="customer_id", logical_type="string", required=True, nullable=False),
            ColumnSchema(name="product_id", logical_type="string", required=True, nullable=False),
            ColumnSchema(name="transaction_date", logical_type="datetime", required=True, nullable=False, expected_format="%Y-%m-%d"),
            ColumnSchema(name="quantity", logical_type="integer", required=True, nullable=False, min_value_exclusive=0),
            ColumnSchema(name="unit_price", logical_type="float", required=True, nullable=False, min_value_exclusive=0.0),
            ColumnSchema(name="total_amount", logical_type="float", required=True, nullable=False, min_value=0.0),
            ColumnSchema(name="region", logical_type="string", required=True, nullable=False, allowed_values=["North", "South", "East", "West", "Central"]),
            ColumnSchema(name="payment_method", logical_type="string", required=True, nullable=False, allowed_values=["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Cash"]),
            ColumnSchema(name="sales_channel", logical_type="string", required=True, nullable=False, allowed_values=["Online", "In-Store", "Phone", "Partner", "Direct Sales"]),
        ],
    )


def load_schema(schema_input: Union[str, Path, Dict[str, Any], SchemaDefinition]) -> SchemaDefinition:
    """Flexible loader for schema input."""
    if isinstance(schema_input, SchemaDefinition):
        return schema_input
    if isinstance(schema_input, dict):
        return SchemaDefinition.from_dict(schema_input)
    if isinstance(schema_input, (str, Path)):
        p = Path(schema_input)
        if p.is_file():
            return SchemaDefinition.from_file(p)
        if str(schema_input).lower() in ("sales", "sales_schema", "sales_schema.json"):
            return get_sales_schema()
        raise FileNotFoundError(f"Schema file or preset '{schema_input}' not found.")
    raise ValueError(f"Unsupported schema input type: {type(schema_input)}")
